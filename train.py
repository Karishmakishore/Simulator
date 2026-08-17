from ultralytics import YOLO

# 1. Load a pretrained model
# If you have a specific .pt file in your folder, point to it
# Otherwise, 'yolov8n.pt' will download automatically to your path
model = YOLO('/home/aniket/Desktop/YOLO_tut/yolov8n.pt')

# 2. Train the model
results = model.train(
    data='/home/aniket/Desktop/YOLO_tut/data.yaml', 
    epochs=100,      # Increase or decrease based on convergence
    imgsz=640,       # Standard YOLO resolution
    batch=16,        # Adjust based on your GPU VRAM (e.g., 8, 16, 32)
    device=0,        # Use 0 for the first CUDA GPU, or 'cpu' if no GPU
    project='/home/aniket/Desktop/YOLO_tut/runs',
    name='simulator_training'
)

# 3. Validate the model
metrics = model.val()

# 4. Export the model to use in your simulator/project
model.export(format='onnx')