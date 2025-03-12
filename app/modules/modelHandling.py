import torch
import os
from typing import Dict, Any, List, Tuple, Optional
import torchvision.transforms as T
import numpy as np

class ModelDetector:
    """
    Flexible object detection model loader and inference handler.
    Supports different model formats and adapts to input/output requirements.
    """
    def __init__(
        self, 
        model_path: str,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.5,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the model detector.
        
        Args:
            model_path: Path to the PyTorch model file (.pt)
            device: Device to run inference on (default: auto-detect)
            confidence_threshold: Minimum confidence score for detections
            class_names: List of class names for the model
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or []
        self.model = None
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Load the model with safeguards for different formats
            self.model = torch.jit.load(self.model_path, map_location=self.device)
        except Exception as e:
            # Fallback to standard PyTorch loading
            try:
                self.model = torch.load(self.model_path, map_location=self.device)
            except Exception as nested_e:
                raise RuntimeError(f"Failed to load model: {e}. Nested error: {nested_e}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Detect model type and set appropriate handlers
        self._detect_model_type()
    
    def _detect_model_type(self):
        """
        Detect the model type and set appropriate pre/post processing.
        This allows for supporting different model architectures dynamically.
        """
        # This is a placeholder for model type detection logic
        # In a real app, you'd check model structure, output format, etc.
        self.model_type = "generic"
        
        # For now, we'll use a simple method to check if it's likely a YOLO model
        if hasattr(self.model, 'names') and isinstance(getattr(self.model, 'names', None), (list, dict)):
            self.model_type = "yolo"
            if not self.class_names:
                self.class_names = self.model.names
    
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply any model-specific preprocessing beyond the standard processing.
        Adapts to different model requirements.
        
        Args:
            input_data: Dictionary containing input tensor and metadata
            
        Returns:
            Processed input ready for the specific model
        """
        # The ImageProcessor has already done basic preprocessing
        # This method allows for model-specific adjustments
        return input_data
    
    def postprocess(self, model_output: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process model outputs into a standardized format.
        Adapts to different model output formats.
        
        Args:
            model_output: Raw output from the model
            metadata: Dictionary containing original image metadata
            
        Returns:
            Dictionary with standardized detection results
        """
        # Default implementation for generic model
        # Override this in subclasses for specific model types
        
        # Format will depend on the model, but we'll create a standard output:
        # - boxes: [N, 4] tensor of bounding boxes in [x1, y1, x2, y2] format
        # - scores: [N] tensor of confidence scores
        # - labels: [N] tensor of class indices
        
        if self.model_type == "yolo":
            return self._process_yolo_output(model_output, metadata)
        
        # Generic fallback processing
        try:
            if isinstance(model_output, tuple):
                # Try to extract boxes, scores, and labels from tuple
                if len(model_output) >= 3:
                    boxes, scores, labels = model_output[:3]
                else:
                    # Handle other tuple formats
                    boxes = model_output[0]
                    # Dummy scores and labels if not provided
                    scores = torch.ones(boxes.shape[0], device=self.device)
                    labels = torch.zeros(boxes.shape[0], device=self.device)
            elif isinstance(model_output, dict):
                # Try to extract from dictionary
                boxes = model_output.get('boxes', None)
                scores = model_output.get('scores', None)
                labels = model_output.get('labels', None)
            elif isinstance(model_output, torch.Tensor):
                # Handle tensor output (common in YOLO and similar models)
                # Assuming format: [x1, y1, x2, y2, confidence, class1, class2, ...]
                if model_output.dim() == 3:
                    model_output = model_output.squeeze(0)  # Remove batch dimension
                
                if model_output.size(1) > 5:  # Has class scores
                    # Get confidence scores and class indices
                    confidence = model_output[:, 4]
                    class_scores, class_indices = torch.max(model_output[:, 5:], dim=1)
                    scores = confidence * class_scores
                    labels = class_indices
                    boxes = model_output[:, :4]
                else:
                    # Simple case: just boxes and scores
                    boxes = model_output[:, :4]
                    scores = model_output[:, 4]
                    labels = torch.zeros_like(scores, dtype=torch.long)
            else:
                # Unknown format - return empty results
                return {
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "scores": torch.zeros(0, device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device)
                }
            
            # Filter by confidence
            if scores is not None:
                mask = scores > self.confidence_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask] if labels is not None else torch.zeros_like(scores, dtype=torch.long)
            
            return {
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            }
        except Exception as e:
            # If parsing fails, return empty results
            print(f"Error processing model output: {e}")
            return {
                "boxes": torch.zeros((0, 4), device=self.device),
                "scores": torch.zeros(0, device=self.device),
                "labels": torch.zeros(0, dtype=torch.long, device=self.device)
            }
    
    def _process_yolo_output(self, output, metadata):
        """Process YOLO-specific output format."""
        # This is a simplified implementation for YOLO-like models
        if isinstance(output, torch.Tensor):
            # Already in tensor format
            predictions = output
        elif isinstance(output, (tuple, list)):
            # Multiple outputs, use the first one
            predictions = output[0]
        else:
            # Unknown format
            return {
                "boxes": torch.zeros((0, 4), device=self.device),
                "scores": torch.zeros(0, device=self.device),
                "labels": torch.zeros(0, dtype=torch.long, device=self.device)
            }
        
        # Filter by confidence
        if predictions.shape[0] > 0:
            mask = predictions[:, 4] > self.confidence_threshold
            predictions = predictions[mask]
        
        if predictions.shape[0] == 0:
            return {
                "boxes": torch.zeros((0, 4), device=self.device),
                "scores": torch.zeros(0, device=self.device),
                "labels": torch.zeros(0, dtype=torch.long, device=self.device)
            }
        
        # Get boxes, scores, and class indices
        boxes = predictions[:, :4]
        confidences = predictions[:, 4]
        
        if predictions.shape[1] > 5:
            # Multiple class scores
            class_scores, class_indices = torch.max(predictions[:, 5:], dim=1)
            scores = confidences * class_scores
            labels = class_indices
        else:
            # Single class
            scores = confidences
            labels = torch.zeros_like(scores, dtype=torch.long)
        
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the processed input data.
        
        Args:
            input_data: Dictionary containing input tensor and metadata
            
        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Get the input tensor
        input_tensor = input_data["input_tensor"]
        
        # Apply any model-specific preprocessing
        processed_input = self.preprocess(input_data)
        
        # Run inference
        with torch.no_grad():
            try:
                output = self.model(input_tensor)
            except Exception as e:
                print(f"Inference error: {e}")
                # Return empty results
                return {
                    "boxes": torch.zeros((0, 4), device=self.device),
                    "scores": torch.zeros(0, device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device)
                }
        
        # Process the output
        results = self.postprocess(output, input_data)
        
        return results
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"Class {class_idx}"