from roboflow import Roboflow
rf = Roboflow(api_key="C5IyjZmvv6bNThZZ7pHH")
project = rf.workspace("ship-dataset-whmf4").project("simulator-2-6pssz")
version = project.version(1)
dataset = version.download("yolov8")