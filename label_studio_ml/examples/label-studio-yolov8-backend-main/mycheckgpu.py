import torch

# Проверка наличия CUDA
if torch.cuda.is_available():
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU instead.")