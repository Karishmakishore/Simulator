import os
from ultralytics import YOLO

# 1. Load your custom trained model
# Replace 'train/weights/best.pt' with the actual path to your best weights 
# Usually: /home/aniket/Desktop/YOLO_tut/runs/detect/simulator_training/weights/best.pt
model_path = '/home/aniket/Desktop/YOLO_tut/yolov8n.pt'
model = YOLO(model_path)

# 2. Define the path to your test images
test_images_path = '/home/aniket/Desktop/YOLO_tut/Simulator-2-1/test/images'

# 3. Run inference on the entire test folder
# 'save=True' will create a new folder in 'runs/detect/predict' with annotated images
# 'conf=0.5' sets the confidence threshold (only show detections > 50%)
results = model.predict(
    source=test_images_path,
    conf=0.25,      # Adjust this to be more or less strict
    save=True,      # Saves the images with bounding boxes
    save_txt=True,  # Saves the prediction coordinates in .txt format
    project='/home/aniket/Desktop/YOLO_tut/test_results',
    name='version_1'
)

print(f"Testing complete. Results saved to: /home/aniket/Desktop/YOLO_tut/test_results/version_1")