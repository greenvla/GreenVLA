from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Dict
from torch.utils.data import Dataset
from lerobot.common.utils.utils import get_local_hf_snapshot_or_repo_id
from transformers import AutoProcessor
import numpy as np
import json
import albumentations as A
import logging
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich import print
import pandas as pd


@dataclass(frozen=True)
class QwenVLMDatasetConfig:
    dataset_id: str
    dataset_path: str
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    image_shape: Tuple[int, int] = (224, 224)
    max_len: int = 512
    image_keys: Optional[List[str]] = None

class QwenVLMDataset(Dataset):
    _truncation_warning_shown = False
    
    def __init__(self, config: QwenVLMDatasetConfig):
        self.config = config
        self.dataset_id = config.dataset_id
        self.dataset_path = config.dataset_path
        self.model_id = config.model_id
        self.image_shape = config.image_shape
        self.max_len = config.max_len
        # default image keys if not provided
        self.image_keys = config.image_keys or [
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ]
        self.processor = AutoProcessor.from_pretrained(get_local_hf_snapshot_or_repo_id(self.model_id), fix_mistral_regex=True)
        with open(self.dataset_path, "r") as f:
            self.dataset = json.load(f)
        self.resize = A.Resize(self.image_shape[1], self.image_shape[0])

    def __len__(self):
        return len(self.dataset)
    
    def compute_image_thw(self, img_height: int, img_width: int):
        grid_t = 1
        grid_h = img_height // self.processor.image_processor.patch_size
        grid_w = img_width // self.processor.image_processor.patch_size
        return np.array([grid_t, grid_h, grid_w])
    
    def __getitem__(self, idx: int):
        
        sample = self.dataset[idx]
        
        if "images" in sample:
            images = []
            num_images = len(sample['images'])
            for img_path in sample['images']:
                img = np.array(Image.open(img_path).convert("RGB"))
                images.append(img)
            
        else:
            raise ValueError(f"No images found in sample {sample}")
        
        assert num_images <= 3, f"Number of images ({num_images}) exceeds the maximum number of images (3)"
        image_dict: Dict[str, torch.Tensor] = {}
        mask_dict: Dict[str, torch.Tensor] = {}
        for i, key in enumerate(self.image_keys):
            if i < num_images:
                image_dict[key] = torch.from_numpy(self.resize(image=images[i])["image"]).permute(2, 0, 1)
                mask_dict[key] = torch.tensor(True)
            else:
                image_dict[key] = torch.zeros((3, *self.image_shape), dtype=torch.uint8)
                mask_dict[key] = torch.tensor(False)

        
       
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy_value"} for _ in range(num_images)
                ]
                + [
                    {"type": "text", "text": sample['question']}
                ],  # keeped here the ordering images; text because original Qwen works better in zero shot with this ordering
            },
        ]
        text_inputs = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        image_grid_thw = [
            self.compute_image_thw(self.image_shape[0], self.image_shape[1])
        ] * num_images
        index = 0
        while self.processor.image_token in text_inputs:
            text_inputs = text_inputs.replace(
                self.processor.image_token,
                "<|placeholder|>"
                * (
                    image_grid_thw[index].prod()
                    // self.processor.image_processor.merge_size**2
                ),
                1,
            )
            index += 1
        
        text_inputs = text_inputs.replace(
            "<|placeholder|>", self.processor.image_token
        )
        # Adding 
        tokenized_text = self.processor.tokenizer(
            text_inputs,
            return_tensors=None,
            padding="longest",
        )
        prefix_tokens = tokenized_text["input_ids"]
        
        postfix_text = "<|im_start|>assistant\n" + sample['answer'] + "<|im_end|>"
        postfix_tokens = self.processor.tokenizer(postfix_text, return_tensors=None, padding="longest")["input_ids"]
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        # 0 - prefix, 1 - not used here, 2 - postfix (with loss), 3 - padding
        token_type_ids = [0] * len(prefix_tokens) + [2] * len(postfix_tokens)
        
                # Pad tokens to max length
        tokens_len = len(tokens)
        
        if tokens_len < self.max_len:
            padding = [self.processor.tokenizer.pad_token_id] * (
                self.max_len - tokens_len
            )
            padding_mask = [False] * (self.max_len - tokens_len)
            token_type_ids = token_type_ids + [3] * len(padding)
            tokens = tokens + padding
            token_mask = token_mask + padding_mask
            loss_mask = loss_mask + padding_mask

        else:
            if len(tokens) > self.max_len and not QwenVLMDataset._truncation_warning_shown:
                QwenVLMDataset._truncation_warning_shown = True
                logging.warning(
                    f"Token length in general domain VLM data ({len(tokens)}) exceeds max length ({self.max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self.max_len]
            token_mask = token_mask[: self.max_len]
            loss_mask = loss_mask[: self.max_len]
            token_type_ids = token_type_ids[: self.max_len]

        
        
        
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "padded_mask": torch.tensor(token_mask, dtype=torch.long),
            "attention_mask": torch.tensor(token_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "image": image_dict,
            "image_mask": mask_dict,
        }
        

@dataclass(frozen=True)
class MixtureQwenVLMDatasetConfig:
    datasets: list[str, Any]
    weights: Optional[dict[str, float]] = None
    

class MixtureQwenVLMDataset(Dataset):
    def __init__(self, config: MixtureQwenVLMDatasetConfig):
        # asserting that weiths are same as datasets
        if config.weights is not None:
            assert len(config.datasets) == len(
                config.weights
            ), "Number of datasets and weights must be the same"
            assert set(config.datasets.keys()) == set(
                config.weights.keys()
            ), f"Datasets and weights must have the same keys; missing keys: {set(config.weights.keys()) - set(config.datasets.keys())}"
        else:
            print("No weights provided, using default weights")
        
        self.datasets = [v for v in config.datasets.values()]
        if config.weights is None:
            weights = [len(v) ** 0.43 for v in self.datasets]
        else:
            weights = [config.weights[key] for key in config.datasets.keys()]

        weights = np.array(weights)
        weights = weights / sum(weights)
        self.weights = weights
        self._length = sum(len(ds) for ds in self.datasets)
        self.rng = np.random.RandomState(42)
        
    def __len__(self):
        return self._length
    
    def set_rng(self, rng):
        self.rng = rng
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        choosed_dataset_idx = self.rng.choice(len(self.datasets), p=self.weights)
        choosed_element_idx = self.rng.randint(
            0, len(self.datasets[choosed_dataset_idx])
        )
        try:
            batch = self.datasets[choosed_dataset_idx][choosed_element_idx]
            # Add dataset_id to match LeRobotMixtureDataset behavior
            if isinstance(batch, dict):
                batch["dataset_id"] = self.datasets[choosed_dataset_idx].dataset_id
            else:
                batch = {"data": batch, "dataset_id": self.datasets[choosed_dataset_idx].dataset_id}
            return batch
        except Exception as e:
            print(
                f"Error in dataset {self.datasets[choosed_dataset_idx].dataset_id} at index {choosed_element_idx}\n{e}"
            )
            return self[idx]

    def print_dataset_summary(self):
        table = Table(title="[bold green]QwenVLMMixtureDataset Summary[/bold green]")
        table.add_column("Dataset ID", style="bold cyan", no_wrap=True)
        table.add_column("Amount of samples", style="bold magenta")
        table.add_column("Dataset weight", style="bold magenta")

        for idx, dataset in enumerate(self.datasets):
            table.add_row(
                dataset.dataset_id,
                str(format(len(dataset), ",")),
                f"{self.weights[idx]:.3f}",
            )

        # adding total amount of samples
        table.add_row("Total", str(format(self._length, ",")), "")

        console = Console()
        console.print(table)
    
    def build_summary_df(self):
        rows = []
        for idx, dataset in enumerate(self.datasets):
            rows.append({
                "dataset_id": dataset.dataset_id,
                "num_samples": len(dataset),
                "weight": self.weights[idx],
            })
        
        rows.append({
            "dataset_id": "Total",
            "num_samples": self._length,
            "weight": "",
        })
        return pd.DataFrame(rows)
    
    def log_dataset_summary(self, logger):
        
        df = self.build_summary_df()
        logger.report_table(
            title="Training Dataset Summary (Global)",
            series="VLM Dataset Summary",
            iteration=0,
            table_plot=df,
        )