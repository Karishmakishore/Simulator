# Simulator project

This repository contains the training scripts and dataset config for the Simulator project.

Important: the Roboflow API key has been removed from the committed files for security. Before running the scripts you must set the environment variable ROBOFLOW_API_KEY with your Roboflow API key.

On macOS / Linux (bash/zsh):

  export ROBOFLOW_API_KEY="your_actual_key_here"

On Windows (PowerShell):

  $env:ROBOFLOW_API_KEY = "your_actual_key_here"

After setting the variable, you can run the training script:

  python train.py

Notes:
- Do NOT commit your API key to the repository. If you previously committed a key to a public repo, rotate it immediately.
- If your dataset or model files are large (>100 MB), consider using Git LFS or storing artifacts outside the repo and adding download instructions instead.
