import torch

print("PyTorch version:", torch.__version__)
print("\nCUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))

print("\nMPS (Metal Performance Shaders) available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    print("MPS device name: Apple Silicon/AMD GPU")

# Test GPU with a simple operation
if torch.cuda.is_available():
    print("\nTesting CUDA with a simple operation:")
    device = torch.device("cuda")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    print("Matrix multiplication completed successfully on CUDA GPU")

if torch.backends.mps.is_available():
    print("\nTesting MPS with a simple operation:")
    device = torch.device("mps")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    print("Matrix multiplication completed successfully on MPS") 