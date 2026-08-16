import os
from roboflow import Roboflow

# Read Roboflow API key from environment variable for security
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("simulator-flvry").project("simulator-2")
version = project.version(3)
dataset = version.download("yolov8")
