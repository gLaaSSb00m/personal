import torch

def check_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU detected or CUDA is not available.")

if __name__ == "__main__":
    check_gpu()