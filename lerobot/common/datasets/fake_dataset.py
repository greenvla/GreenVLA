import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Tuple, Optional

class FakeDataset(Dataset):
    """
    A fake dataset that generates random data for testing and debugging purposes.
    It aims to produce data in a structure similar to what LeRobot policies might expect.
    """
    def __init__(
        self,
        model_config: Any, # Can be a config object or a placeholder
        num_samples: int = 1024,
        image_size: Tuple[int, int] = (224, 224), # H, W
        num_cameras: int = 3,
        state_dim: int = 14, # Example state dimension
        # model_config can provide action_dim and action_horizon
    ):
        self._num_samples = num_samples
        self.image_height, self.image_width = image_size
        self.num_cameras = num_cameras
        self.state_dim = state_dim

        # Try to get action_dim and action_horizon from model_config
        # These attributes are present in LeRobotModelConfigPlaceholder
        self.action_dim = getattr(model_config, 'max_action_dim', 14)
        self.action_horizon = getattr(model_config, 'action_horizon', 16)
        
        # Placeholder for more detailed specs from model_config if available
        self.observation_spec = getattr(model_config, 'observation_spec', None)
        self.action_spec = getattr(model_config, 'action_spec', None)

    def __len__(self) -> int:
        return self._num_samples

    def _generate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        if dtype.is_floating_point:
            return torch.rand(shape, dtype=dtype) * (high - low) + low
        elif dtype == torch.uint8:
            return torch.randint(int(low), int(high) + 1, shape, dtype=dtype)
        elif dtype == torch.int32 or dtype == torch.int64:
             return torch.randint(int(low), int(high) + 1, shape, dtype=dtype)
        else:
            return torch.zeros(shape, dtype=dtype)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < self._num_samples:
            raise IndexError("Index out of bounds")

        sample: Dict[str, Any] = {
            'observation': {
                'images': {},
                'state': None,
            },
            'action': None, # For compatibility with some policy inputs
            'actions': None, # Often used for training
            'episode_index': torch.tensor(index % 10, dtype=torch.int64), # Example episode index
            'prompt': f"Fake task for sample {index}" # Example prompt
            # Add other keys like 'timestamp', 'frame_id' if commonly needed
        }

        # Generate image data
        for i in range(self.num_cameras):
            cam_name = f'camera_{i}' # e.g. camera_0, camera_1, camera_2
            # Default image format: (C, H, W), float32
            # Some pipelines might expect (H, W, C)
            img_shape = (3, self.image_height, self.image_width)
            img_dtype = torch.float32
            
            # Override with spec if available
            if self.observation_spec and f'observation/images/{cam_name}' in self.observation_spec:
                spec = self.observation_spec[f'observation/images/{cam_name}']
                img_shape = spec.get('shape', img_shape)
                img_dtype = spec.get('dtype', img_dtype)

            sample['observation']['images'][cam_name] = self._generate_tensor(img_shape, img_dtype)

        # Generate state data
        state_shape = (self.state_dim,)
        state_dtype = torch.float32
        if self.observation_spec and 'observation/state' in self.observation_spec:
            spec = self.observation_spec['observation/state']
            state_shape = spec.get('shape', state_shape)
            state_dtype = spec.get('dtype', state_dtype)
        sample['observation']['state'] = self._generate_tensor(state_shape, state_dtype)

        # Generate action data
        # This usually corresponds to 'actions' key for training Diffusion Policy like models
        action_shape = (self.action_horizon, self.action_dim)
        action_dtype = torch.float32
        if self.action_spec: # Assuming action_spec directly defines the 'actions' tensor
            # Action spec might be simpler, e.g. just for a single action tensor
            # Adapt if action_spec is a tree
            spec = self.action_spec.get('actions', self.action_spec) # Try 'actions' key or root of spec
            action_shape = spec.get('shape', action_shape)
            action_dtype = spec.get('dtype', action_dtype)
            
        sample['actions'] = self._generate_tensor(action_shape, action_dtype)
        
        # Often, policies take a single 'action' key as well during inference,
        # which could be the first step of the 'actions' horizon.
        sample['action'] = sample['actions'][0].clone() if self.action_horizon > 0 else self._generate_tensor((self.action_dim,), action_dtype)


        # The LeRobotDataset returns a flat dictionary.
        # We need to flatten the sample dict to match that.
        flat_sample = {}
        def _flatten_recursively(d, parent_key='', sep='/'):
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict): # Check if it's a dict that should be flattened
                    _flatten_recursively(v, new_key, sep=sep)
                else:
                    flat_sample[new_key] = v
        
        _flatten_recursively(sample)
        
        # Ensure all expected top-level keys by LeRobot are present, even if None
        # Example: 'keypoint_annotations', 'instruction', etc.
        # For now, this simplified FakeDataset might not produce all of them.
        # Users of this FakeDataset should be aware of its limitations.

        return flat_sample

if __name__ == '__main__':
    # Example Usage:
    class MockModelConfig:
        def __init__(self):
            self.max_action_dim = 7
            self.action_horizon = 10
            # Example of providing detailed specs (optional)
            self.observation_spec = {
                "observation/images/camera_0": {"shape": (3, 64, 64), "dtype": torch.float32},
                "observation/state": {"shape": (5,), "dtype": torch.float32}
            }
            self.action_spec = { # For the main 'actions' tensor
                "shape": (10, 7), # horizon, dim
                "dtype": torch.float32
            }


    mock_config = MockModelConfig()
    fake_dataset = FakeDataset(model_config=mock_config, num_samples=10, num_cameras=1, state_dim=5)
    print(f"Fake dataset length: {len(fake_dataset)}")
    
    sample_0 = fake_dataset[0]
    print("\nSample 0 keys:", sample_0.keys())
    for key, value in sample_0.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {value} (type {type(value)})")

    # Example with default config (less specific)
    class BasicConfig:
        max_action_dim = 14
        action_horizon = 16
        
    basic_config = BasicConfig()
    default_fake_dataset = FakeDataset(model_config=basic_config, num_samples=5)
    sample_default = default_fake_dataset[0]
    print("\nDefault Sample keys:", sample_default.keys())
    for key, value in sample_default.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
