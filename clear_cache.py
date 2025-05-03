import torch
import gc

def clear_gpu_cache():
    if torch.cuda.is_available():
        print("Clearing GPU memory and cache...")
        # Delete unused tensors
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("GPU memory and cache cleared.")
    else:
        print("No GPU detected. Nothing to clear.")

if __name__ == "__main__":
    clear_gpu_cache()