import cv2
import torch
from typing import Optional, Tuple
import platform
import time

class CameraModule:
    """
    A cross-platform camera module that captures frames for object detection.
    Works on laptop, mobile, or other devices with a camera.
    """
    def __init__(self, camera_id: int = 0, img_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the camera module.
        
        Args:
            camera_id: Camera device ID (default: 0 for primary camera)
            img_size: Output image dimensions as (width, height)
        """
        self.camera_id = camera_id
        self.img_size = img_size
        self.camera = None
        self.is_running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def check_camera(self) -> bool:
        """Check if the specified camera is available."""
        try:
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                return False
            cap.release()
            return True
        except Exception:
            return False
            
    def get_available_cameras(self) -> list:
        """Detect available camera devices on the system."""
        available_cameras = []
        
        # Different max camera check based on platform
        max_to_check = 1
        if platform.system() == 'Windows' or platform.system() == 'Linux':
            max_to_check = 5  # Check more cameras on desktop systems
            
        for i in range(max_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
                
        return available_cameras
            
    def start(self) -> bool:
        """Start the camera capture."""
        if self.is_running:
            return True
            
        if not self.check_camera():
            available = self.get_available_cameras()
            if not available:
                return False
            self.camera_id = available[0]  # Use first available camera
            
        self.camera = cv2.VideoCapture(self.camera_id)
        
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])
        
        self.is_running = self.camera.isOpened()
        return self.is_running
        
    def get_frame(self) -> Optional[torch.Tensor]:
        """
        Capture a frame from the camera and convert to PyTorch tensor.
        
        Returns:
            Tensor of shape [3, height, width] or None if capture failed
        """
        if not self.is_running:
            if not self.start():
                return None
                
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch tensor and normalize to [0, 1]
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        
        # Resize if necessary
        if frame_rgb.shape[1] != self.img_size[0] or frame_rgb.shape[0] != self.img_size[1]:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), 
                size=(self.img_size[1], self.img_size[0]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
        return tensor.to(self.device)
        
    def stop(self):
        """Stop camera capture and release resources."""
        if self.camera and self.is_running:
            self.camera.release()
            self.is_running = False
            
    def __del__(self):
        """Ensure camera resources are released."""
        self.stop()