import os
import wandb
import torch
import torch.distributed as dist
from ultralytics import YOLO
from huggingface_hub import HfApi, Repository, login
import shutil
import sys
import subprocess
import psutil
import signal

# Keep existing HF settings and helper functions...
HF_REPO = "EduTech-YOLOv11"
HF_USERNAME = "shng2025"

def ensure_hf_login():
    """Ensure user is logged in to Hugging Face"""
    try:
        api = HfApi()
        # This will raise an error if not logged in
        api.whoami()
        print("Already logged in to Hugging Face")
    except Exception:
        print("Please log in to Hugging Face")
        login()

def run_git_command(command, cwd=None):
    """Run a git command and return its output"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

# Add DDP setup function
def setup_ddp():
    """Setup distributed data parallel training"""
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '4'  # 4 GPUs
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    # Initialize the process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def ensure_git_lfs():
    """Ensure git-lfs is installed and initialized"""
    try:
        # Check if git-lfs is installed
        result = subprocess.run(['git-lfs', '--version'], 
                              capture_output=True, 
                              text=True)
        print("git-lfs is already installed:", result.stdout.strip())
    except FileNotFoundError:
        print("git-lfs not found. Attempting to install...")
        
        # Try to install git-lfs based on the system
        try:
            # First, update package list
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            # Install git-lfs
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'git-lfs'], check=True)
            # Initialize git-lfs
            subprocess.run(['git', 'lfs', 'install'], check=True)
            print("git-lfs installed and initialized successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing git-lfs: {e}")
            print("Please install git-lfs manually:")
            print("1. Run: sudo apt-get update")
            print("2. Run: sudo apt-get install git-lfs")
            print("3. Run: git lfs install")
            sys.exit(1)

def setup_hf_repo(branch_name):
    """Setup or connect to Hugging Face repository"""
    # Ensure git-lfs is installed first
    ensure_git_lfs()
    
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_REPO}"
    repo_url = f"https://huggingface.co/{repo_id}"
    local_dir = os.path.abspath(HF_REPO)
    
    # First, check if the repo exists on HuggingFace
    try:
        api.repo_info(repo_id)
        print(f"Repository {repo_id} found on HuggingFace")
    except Exception:
        print(f"Repository {repo_id} not found. Creating new repository...")
        api.create_repo(repo_id, private=True)
    
    # Remove local directory if it exists but is not a valid git repo
    if os.path.exists(local_dir):
        try:
            Repository(local_dir=local_dir)
        except ValueError:
            print(f"Removing invalid repository at {local_dir}")
            shutil.rmtree(local_dir)
    
    # Clone or use existing repository
    if not os.path.exists(local_dir):
        print(f"Cloning repository from {repo_url}...")
        repo = Repository(
            local_dir=local_dir,
            clone_from=repo_url,
            use_auth_token=True
        )
    else:
        print(f"Using existing repository at {local_dir}")
        repo = Repository(
            local_dir=local_dir,
            use_auth_token=True
        )
    
    # Initialize git lfs
    print("Initializing Git LFS...")
    run_git_command("git lfs install", cwd=local_dir)
    
    # Create and checkout new branch or checkout existing branch
    current_branch = run_git_command("git rev-parse --abbrev-ref HEAD", cwd=local_dir)
    if current_branch and current_branch != branch_name:
        print(f"Switching from branch {current_branch} to {branch_name}")
        run_git_command(f"git checkout -B {branch_name}", cwd=local_dir)
    
    # Initialize git lfs tracking for specific file types
    run_git_command("git lfs track '*.pt'", cwd=local_dir)  # Track PyTorch model files
    run_git_command("git lfs track '*.pth'", cwd=local_dir)  # Track PyTorch state dictionaries
    run_git_command("git lfs track '*.bin'", cwd=local_dir)  # Track binary files
    
    # Add .gitattributes to track LFS configurations
    gitattributes_path = os.path.join(local_dir, '.gitattributes')
    if os.path.exists(gitattributes_path):
        run_git_command("git add .gitattributes", cwd=local_dir)
        try:
            run_git_command('git commit -m "Update LFS tracking configurations"', cwd=local_dir)
        except Exception:
            print("No changes to .gitattributes to commit")
    
    print(f"Successfully set up repository and switched to branch: {branch_name}")
    return repo

def setup_hf_repo(branch_name):
    """Setup or connect to Hugging Face repository"""
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_REPO}"
    repo_url = f"https://huggingface.co/{repo_id}"
    local_dir = os.path.abspath(HF_REPO)
    
    # First, check if the repo exists on HuggingFace
    try:
        api.repo_info(repo_id)
        print(f"Repository {repo_id} found on HuggingFace")
    except Exception:
        print(f"Repository {repo_id} not found. Creating new repository...")
        api.create_repo(repo_id, private=True)
    
    # Remove local directory if it exists but is not a valid git repo
    if os.path.exists(local_dir):
        try:
            Repository(local_dir=local_dir)
        except ValueError:
            print(f"Removing invalid repository at {local_dir}")
            shutil.rmtree(local_dir)
    
    # Clone or use existing repository
    if not os.path.exists(local_dir):
        print(f"Cloning repository from {repo_url}...")
        repo = Repository(
            local_dir=local_dir,
            clone_from=repo_url,
            use_auth_token=True
        )
    else:
        print(f"Using existing repository at {local_dir}")
        repo = Repository(
            local_dir=local_dir,
            use_auth_token=True
        )
    
    # Initialize git lfs
    print("Initializing Git LFS...")
    run_git_command("git lfs install", cwd=local_dir)
    
    # Create and checkout new branch or checkout existing branch
    current_branch = run_git_command("git rev-parse --abbrev-ref HEAD", cwd=local_dir)
    if current_branch and current_branch != branch_name:
        print(f"Switching from branch {current_branch} to {branch_name}")
        run_git_command(f"git checkout -B {branch_name}", cwd=local_dir)
    
    # Initialize git lfs tracking for specific file types
    run_git_command("git lfs track '*.pt'", cwd=local_dir)  # Track PyTorch model files
    run_git_command("git lfs track '*.pth'", cwd=local_dir)  # Track PyTorch state dictionaries
    run_git_command("git lfs track '*.bin'", cwd=local_dir)  # Track binary files
    
    # Add .gitattributes to track LFS configurations
    gitattributes_path = os.path.join(local_dir, '.gitattributes')
    if os.path.exists(gitattributes_path):
        run_git_command("git add .gitattributes", cwd=local_dir)
        try:
            run_git_command('git commit -m "Update LFS tracking configurations"', cwd=local_dir)
        except Exception:
            print("No changes to .gitattributes to commit")
    
    print(f"Successfully set up repository and switched to branch: {branch_name}")
    return repo

def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# Keep all your existing helper functions (ensure_hf_login, run_git_command, etc.)
# ... (copy them from your original code)

# Get environment variables for distributed training
RANK = int(os.environ.get('RANK', -1))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

def check_nvidia_drivers():
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print(f"NVIDIA-SMI output:\n{nvidia_smi}")
        
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA version: {cuda_version}")
        
        nccl_version = torch.cuda.nccl.version()
        print(f"NCCL version: {nccl_version}")
    except Exception as e:
        print(f"Error checking NVIDIA setup: {e}")

def init_distributed():
    try:
        # Set CUDA device before initializing process group
        torch.cuda.set_device(LOCAL_RANK)
        
        # Initialize process group with additional timeout
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=datetime.timedelta(minutes=60)
        )
        
        # Verify initialization
        if dist.is_initialized():
            print(f"Process group initialized successfully on rank {RANK}")
        else:
            print(f"Failed to initialize process group on rank {RANK}")
            
    except Exception as e:
        print(f"Error initializing distributed setup on rank {RANK}: {e}")
        raise

def main():
    print(f"\n{'-'*40}")
    print(f"Starting process: {RANK=} -- {LOCAL_RANK=} -- {WORLD_SIZE=}")
    
    # Check NVIDIA setup
    if RANK == 0:
        check_nvidia_drivers()
    
    try:
        # Initialize distributed setup
        init_distributed()
        
        # Initialize wandb only on rank 0
        if RANK == 0:
            wandb.init(project="Edutech")
            run_name = wandb.run.name
        
        # Set CUDA device
        device = torch.device(f'cuda:{LOCAL_RANK}')
        print(f"Rank {RANK} using device: {device}")
        
        # Load model
        print(f"About to load model on rank {RANK}")
        model = YOLO("yolo11l.pt")
        
        print(f"Starting training on rank {RANK}")
        start = time.perf_counter()
        
        results = model.train(
            task="detect",
            data="combined_dataset/data.yaml",
            epochs=100,
            batch=120,
            imgsz=640,
            project='Edutech',
            name=run_name if RANK == 0 else None,
            device=[LOCAL_RANK],
            exist_ok=True,
            amp=True,
            plots=True if RANK == 0 else False,
            save=True if RANK == 0 else False,
            close_mosaic=10,
            patience=50,
            save_period=10,
        )
        
        print(f"Training took {time.perf_counter() - start:.5f} seconds to run")
        
        # Validation and export only on rank 0
        if RANK == 0:
            success = model.val()
            if success:
                model.export()
            wandb.finish()
        
    except Exception as e:
        print(f"Error on rank {RANK}: {e}")
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    # Set environment variables for NCCL
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    main()