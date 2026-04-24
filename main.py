# Using transparent-background library
# https://github.com/plemeri/transparent-background
import json
import time
from PIL import Image
from transparent_background import Remover

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Load model configuration
model_cfg = config["model"]

# Initialize remover
remover = Remover(
    mode=model_cfg["mode"],
    jit=model_cfg["jit"],
    device=model_cfg["device"],
    resize=model_cfg["resize"]
)

# Load input image
img_path = config["input_image"]
img = Image.open(img_path).convert("RGB")

# Measure processing time
start_time = time.perf_counter()

# Load process configuration
process_cfg = config["process"]

# Process image
out = remover.process(
    img,
    type=process_cfg["type"],
    threshold=process_cfg["threshold"],
    reverse=process_cfg["reverse"]
)

# Measure processing time
end_time = time.perf_counter()
process_time = end_time - start_time

# Save output
out.save(config["output_image"])

# Output Log
print(f"Done!")
print(f"Output saved to: {config['output_image']}")
print(f"Processing time: {process_time:.4f} seconds")