import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from registry.registry import MODELS

@MODELS.register_module()
class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction pass."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """Load model."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 