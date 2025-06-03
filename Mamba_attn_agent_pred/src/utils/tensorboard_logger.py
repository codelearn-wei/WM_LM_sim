import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class TensorBoardLogger:
    def __init__(self, log_dir='runs'):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        # Create timestamp-based log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, f'run_{timestamp}')
        self.writer = SummaryWriter(self.log_dir)
        
    def log_metrics(self, metrics, step, prefix=''):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            prefix: Prefix for metric names
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{prefix}{name}', value, step)
            elif isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'{prefix}{name}', value.item(), step)
    
    def log_model_graph(self, model, input_tensor):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor
        """
        self.writer.add_graph(model, input_tensor)
    
    def log_histogram(self, name, values, step):
        """
        Log histogram of values to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Current training step
        """
        self.writer.add_histogram(name, values, step)
    
    def log_images(self, name, images, step):
        """
        Log images to TensorBoard.
        
        Args:
            name: Name of the image
            images: Batch of images to log
            step: Current training step
        """
        self.writer.add_images(name, images, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close() 