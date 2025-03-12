import torch
import time
from modules.camCapture import CameraModule
from modules.imgProcessing import ImageProcessor

def main():
    """Main application entry point."""
    # Initialize components
    camera = CameraModule()  # Use native resolution
    processor = ImageProcessor(input_size=(640, 640))
    
    print(f"Using device: {camera.device}")
    print(f"Camera available: {camera.check_camera()}")
    
    # Start camera
    if not camera.start():
        available = camera.get_available_cameras()
        if not available:
            print("No cameras found!")
            return
        print(f"Using available camera ID: {available[0]}")
        camera.camera_id = available[0]
        camera.start()
    
    try:
        print("Press Ctrl+C to stop")
        # Main processing loop
        while True:
            # Get frame from camera
            frame = camera.get_frame()
            if frame is None:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Process frame for model
            processed = processor.process(frame)
            
            # Here we would pass the processed frame to the model
            # For now, just print tensor shape
            input_tensor = processed["input_tensor"]
            print(f"Frame processed: {input_tensor.shape}, "
                  f"Original: {processed['original_size']}, "
                  f"Input: {processed['input_size']}")
            
            # Simulate model processing time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up resources
        camera.stop()
        print("Camera stopped")

if __name__ == "__main__":
    main()