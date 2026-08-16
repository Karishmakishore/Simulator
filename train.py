from ultralytics import YOLO
from roboflow import Roboflow
import os

# Read Roboflow API key from environment variable for security
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("simulator-flvry").project("simulator-2")
dataset = project.version(3).download("yolov8")
print("✅ Dataset downloaded at:", dataset.location)
model = YOLO('/Users/karishmakishore/Desktop/Sim/yolov8n.pt')

# 2. Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml", 
    epochs=100,      # Increase or decrease based on convergence
    imgsz=640,       # Standard YOLO resolution
    batch=16,        # Adjust based on your GPU VRAM (e.g., 8, 16, 32)
    device="cpu",        # Use 0 for the first CUDA GPU, or 'cpu' if no GPU
    project='/Users/karishmakishore/Desktop/Sim/runs',
    name='simulator_training'
)
print("🎉 Training Finished Successfully!")
# 3. Validate the model
metrics = model.val()

# 4. Export the model to use in your simulator/project
model.export(format='onnx')
