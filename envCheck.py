import torch
print(torch.cuda.is_available())   # True if CUDA (GPU) is usable by PyTorch
print(torch.cuda.device_count())   # Number of GPUs detected
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # GPU name