import sys
import platform
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def main():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check CUDA
    print("\nChecking CUDA:")
    print(run_command("nvidia-smi"))
    
    # Try importing key packages
    try:
        import tensorflow as tf
        print("\nTensorFlow version:", tf.__version__)
        print("TensorFlow GPU devices:", tf.config.list_physical_devices('GPU'))
    except Exception as e:
        print("\nTensorFlow import error:", str(e))
    
    try:
        import torch
        print("\nPyTorch version:", torch.__version__)
        print("PyTorch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("PyTorch CUDA version:", torch.version.cuda)
    except Exception as e:
        print("\nPyTorch import error:", str(e))

if __name__ == "__main__":
    main()