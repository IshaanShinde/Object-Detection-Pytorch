import torch
import torchvision.transforms as T
from typing import Tuple, Dict, Any

class ImageProcessor:
    """
    Processes images for object detection models.
    Handles normalization, resizing, and format conversion.
    """
    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize the image processor.
        
        Args:
            input_size: Model input dimensions (width, height)
            mean: Normalization mean values for RGB channels
            std: Normalization standard deviation values for RGB channels
        """
        self.input_size = input_size
        self.mean = mean
        self.std = std
        
        # Create transform pipeline
        self.transform = T.Compose([
            T.Resize(self.input_size),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        
    def process(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Process an image for inference.
        
        Args:
            image: PyTorch tensor of shape [3, H, W] with values in [0, 1]
            
        Returns:
            Dictionary containing processed image and metadata
        """
        # Store original dimensions for bounding box scaling later
        orig_h, orig_w = image.shape[1], image.shape[2]
        
        # Apply transformations
        processed = self.transform(image)
        
        # Create batch dimension [1, 3, H, W]
        batched = processed.unsqueeze(0)
        
        return {
            "input_tensor": batched,
            "original_image": image,
            "original_size": (orig_h, orig_w),
            "input_size": self.input_size
        }
    
    def revert_boxes(self, boxes: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Convert bounding boxes from model output space back to original image space.
        
        Args:
            boxes: Tensor of bounding boxes [N, 4] in format [x1, y1, x2, y2]
            metadata: Dictionary containing image metadata from process()
            
        Returns:
            Tensor of boxes scaled to original image dimensions
        """
        orig_h, orig_w = metadata["original_size"]
        input_h, input_w = metadata["input_size"]
        
        # Scale factors
        scale_w = orig_w / input_w
        scale_h = orig_h / input_h
        
        # Apply scaling: [x1, y1, x2, y2] * [scale_w, scale_h, scale_w, scale_h]
        scaled_boxes = boxes.clone()
        scaled_boxes[:, 0] *= scale_w
        scaled_boxes[:, 1] *= scale_h
        scaled_boxes[:, 2] *= scale_w
        scaled_boxes[:, 3] *= scale_h
        
        return scaled_boxes