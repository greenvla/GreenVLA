from torch.utils.data import Dataset
import numpy as np
import torch

class CoTrainDataset(Dataset):
    def __init__(self, robotics_dataset: Dataset, vlm_dataset: Dataset, robotics_prob: float = 0.5, state_dim: int = 32, action_horizon: int = 50):
        self.robotics_dataset = robotics_dataset
        self.vlm_dataset = vlm_dataset
        self.robotics_prob = robotics_prob
        self.rng = np.random.RandomState(42)
        self.state_dim = state_dim
        self.action_horizon = action_horizon
    def __len__(self):
        return len(self.robotics_dataset) + len(self.vlm_dataset)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for CoTrainDataset of length {len(self)}")
        
        
        if self.rng.random() < self.robotics_prob:
            sampled_idx = self.rng.randint(0, len(self.robotics_dataset))
            sample = self.robotics_dataset[sampled_idx]
            #deleting state
            # sample.pop("state")
            # sample.pop("actions")
            # sample["state"]
            data_source = torch.tensor(0, dtype=torch.long)
        else:
            sampled_idx = self.rng.randint(0, len(self.vlm_dataset))
            sample =  self.vlm_dataset[sampled_idx]
            data_source = torch.tensor(1, dtype=torch.long)
            sample["state"] = torch.zeros(self.state_dim)
            sample["actions"] = torch.zeros(self.action_horizon, self.state_dim)
            sample["action_loss_mask"] = torch.zeros(self.state_dim).to(torch.bool)
            
        sample["data_source"] = data_source # 0 is robotics, 1 is vlm
        return sample
    
    def print_dataset_summary(self):
        self.robotics_dataset.print_dataset_summary()
        self.vlm_dataset.print_dataset_summary()
        
    def set_rng(self, rng):
        if hasattr(self.robotics_dataset, "set_rng"):
            self.robotics_dataset.set_rng(rng)
        self.vlm_dataset.set_rng(rng)
        
        
    def log_dataset_summary(self, logger):
        self.robotics_dataset.log_dataset_summary(logger)
        self.vlm_dataset.log_dataset_summary(logger)
        
        
        