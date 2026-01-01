# Use for YOLO (thinhdv env)
import torch
from ultralytics import YOLO
from ptflops import get_model_complexity_info

# Load model
model = YOLO("E:/thanh/ntu_group/thinh/ObjectDetection/yolo11/ultralytics/runs/detect/k4_69.4/weights/best.pt")  # Replace with your .pt file
pytorch_model = model.model

# Compute FLOPs and parameters
macs, params = get_model_complexity_info(pytorch_model, (3, 640, 640), as_strings=True, print_per_layer_stat=False)

print(f"Computational complexity: {macs}")
print(f"Number of parameters: {params}")

""" 1 MAC = 2 FLOPs"""
"""USE fvcore to calculate GFLOPS AND PARAMS"""
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# Load model
model = YOLO("E:/thanh/ntu_group/thinh/ObjectDetection/yolo11/ultralytics/runs/detect/k4_69.4/weights/best.pt")  # Replace with your .pt file
pytorch_model = model.model

# Define input
inputs = torch.randn(1, 3, 640, 640)

# Calculate FLOPs
flops = FlopCountAnalysis(pytorch_model, inputs)
params = parameter_count_table(pytorch_model)

# Output results
print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # Convert to GFLOPs
print(params)
