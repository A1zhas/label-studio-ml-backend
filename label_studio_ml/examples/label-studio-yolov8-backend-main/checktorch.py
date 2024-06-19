try:
    import torch
    print("PyTorch is installed.")
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed.")