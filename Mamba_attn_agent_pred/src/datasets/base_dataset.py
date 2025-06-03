import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, Optional
from src.registry.registry import DATASETS

@DATASETS.register_module()
class BaseDataset(Dataset):
    """Base class for all datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self.transform = None
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample."""
        raise NotImplementedError
    
    def load_data(self):
        """Load data from disk."""
        raise NotImplementedError
    
    def preprocess(self):
        """Preprocess the data."""
        raise NotImplementedError
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'num_samples': len(self),
            'config': self.config
        } 