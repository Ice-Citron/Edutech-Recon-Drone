import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import torch
from ultralytics import YOLO
from huggingface_hub import HfApi, Repository, login
import shutil
import sys
import subprocess

# Hugging Face settings
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

def find_latest_checkpoint(repo_path, branch_name):
    """Find the latest checkpoint in the repository"""
    # Use os.path.join to create the full path
    local_dir = os.path.join(os.getcwd(), repo_path)
    checkpoint_dir = os.path.join(local_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return None, 0
    
    epochs = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("epoch_"):
            try:
                epoch_num = int(d.split("_")[1])
                epochs.append(epoch_num)
            except ValueError:
                continue
    
    if not epochs:
        print("No checkpoints found")
        return None, 0
    
    latest_epoch = max(epochs)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{latest_epoch}", f"model_epoch_{latest_epoch}.pt")
    print(f"Found latest checkpoint at: {checkpoint_path}")
    return checkpoint_path, latest_epoch

def setup_hf_repo(branch_name):
    """Setup or connect to existing Hugging Face repository"""
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_REPO}"
    
    # Use existing repository directory
    local_dir = os.path.join(os.getcwd(), "EduTech-YOLOv11")
    
    if not os.path.exists(local_dir):
        raise ValueError(f"Repository directory {local_dir} does not exist!")
    
    print(f"Using existing repository at {local_dir}")
    repo = Repository(local_dir=local_dir, use_auth_token=True)
    
    # Initialize git lfs
    print("Initializing Git LFS...")
    run_git_command("git lfs install", cwd=local_dir)
    
    # Create and checkout new branch or checkout existing branch
    current_branch = run_git_command("git rev-parse --abbrev-ref HEAD", cwd=local_dir)
    if current_branch and current_branch != branch_name:
        print(f"Switching from branch {current_branch} to {branch_name}")
        run_git_command(f"git checkout -B {branch_name}", cwd=local_dir)
    
    # Initialize git lfs tracking for specific file types
    run_git_command("git lfs track '*.pt'", cwd=local_dir)
    run_git_command("git lfs track '*.pth'", cwd=local_dir)
    run_git_command("git lfs track '*.bin'", cwd=local_dir)
    
    # Add .gitattributes only if there are changes
    gitattributes_path = os.path.join(local_dir, '.gitattributes')
    if os.path.exists(gitattributes_path):
        # Check if there are any changes to commit
        status = run_git_command("git status --porcelain .gitattributes", cwd=local_dir)
        if status:  # Only commit if there are changes
            run_git_command("git add .gitattributes", cwd=local_dir)
            try:
                run_git_command('git commit -m "Update LFS tracking configurations"', cwd=local_dir)
            except subprocess.CalledProcessError:
                print("No changes to commit for .gitattributes")
    
    return repo, local_dir

def save_checkpoint(model, epoch, repo, branch_name):
    """Save checkpoint to Hugging Face repository"""
    try:
        local_dir = repo.local_dir
        checkpoint_dir = os.path.join(local_dir, "checkpoints", f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure we're on the correct branch
        current_branch = run_git_command("git rev-parse --abbrev-ref HEAD", cwd=local_dir)
        if current_branch != branch_name:
            print(f"Switching to branch {branch_name} before saving checkpoint...")
            run_git_command(f"git checkout {branch_name}", cwd=local_dir)
        
        # Configure git user if not set
        if not run_git_command("git config user.email", cwd=local_dir):
            run_git_command('git config user.email "huggingface@example.com"', cwd=local_dir)
            run_git_command('git config user.name "HuggingFace"', cwd=local_dir)
        
        # Save model state
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        
        # Handle compiled model
        if model.model.__class__.__name__ == 'OptimizedModule':
            # If model is compiled, get the original model for saving
            save_model = model.model._orig_mod
        else:
            save_model = model.model
            
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': save_model.state_dict(),
            'wandb_run_id': wandb.run.id if wandb.run else None
        }
        
        print(f"Saving checkpoint to {checkpoint_path}...")
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest version
        latest_path = os.path.join(local_dir, "latest_model.pt")
        shutil.copy(checkpoint_path, latest_path)
        
        # Pull before pushing to avoid conflicts
        print("Pulling latest changes...")
        run_git_command("git pull origin " + branch_name, cwd=local_dir)
        
        # Add files to git LFS tracking if needed
        run_git_command("git lfs track '*.pt'", cwd=local_dir)
        
        # Commit and push changes
        print("Committing and pushing checkpoint...")
        run_git_command("git add .", cwd=local_dir)
        try:
            run_git_command(f'git commit -m "Checkpoint at epoch {epoch}"', cwd=local_dir)
            run_git_command(f"git push origin {branch_name}", cwd=local_dir)
            print(f"Successfully saved and pushed checkpoint for epoch {epoch}")
        except subprocess.CalledProcessError as e:
            if "nothing to commit" in str(e.stderr):
                print("No changes to commit")
            else:
                raise
                
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing training despite checkpoint save failure...")
        return False
        
    return True

def find_latest_checkpoint(repo_path, branch_name):
    """Find the latest checkpoint in the repository"""
    # Use os.path.join to create the full path
    local_dir = os.path.join(os.getcwd(), repo_path)
    checkpoint_dir = os.path.join(local_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return None, 0
    
    epochs = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("epoch_"):
            try:
                epoch_num = int(d.split("_")[1])
                epochs.append(epoch_num)
            except ValueError:
                continue
    
    if not epochs:
        print("No checkpoints found")
        return None, 0
    
    latest_epoch = max(epochs)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{latest_epoch}", f"model_epoch_{latest_epoch}.pt")
    print(f"Found latest checkpoint at: {checkpoint_path}")
    return checkpoint_path, latest_epoch

def setup_hf_repo(branch_name):
    """Setup or connect to existing Hugging Face repository"""
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_REPO}"
    
    # Use existing repository directory
    local_dir = os.path.join(os.getcwd(), "EduTech-YOLOv11")
    
    if not os.path.exists(local_dir):
        raise ValueError(f"Repository directory {local_dir} does not exist!")
    
    print(f"Using existing repository at {local_dir}")
    repo = Repository(local_dir=local_dir, use_auth_token=True)
    
    # Initialize git lfs
    print("Initializing Git LFS...")
    run_git_command("git lfs install", cwd=local_dir)
    
    # Create and checkout new branch or checkout existing branch
    current_branch = run_git_command("git rev-parse --abbrev-ref HEAD", cwd=local_dir)
    if current_branch and current_branch != branch_name:
        print(f"Switching from branch {current_branch} to {branch_name}")
        run_git_command(f"git checkout -B {branch_name}", cwd=local_dir)
    
    # Initialize git lfs tracking for specific file types
    run_git_command("git lfs track '*.pt'", cwd=local_dir)
    run_git_command("git lfs track '*.pth'", cwd=local_dir)
    run_git_command("git lfs track '*.bin'", cwd=local_dir)
    
    # Add .gitattributes only if there are changes
    gitattributes_path = os.path.join(local_dir, '.gitattributes')
    if os.path.exists(gitattributes_path):
        # Check if there are any changes to commit
        status = run_git_command("git status --porcelain .gitattributes", cwd=local_dir)
        if status:  # Only commit if there are changes
            run_git_command("git add .gitattributes", cwd=local_dir)
            try:
                run_git_command('git commit -m "Update LFS tracking configurations"', cwd=local_dir)
            except subprocess.CalledProcessError:
                print("No changes to commit for .gitattributes")
    
    return repo, local_dir

def get_available_branches():
    """Get list of available branches from Hugging Face repo"""
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_REPO}"
    try:
        branches = api.list_repo_refs(repo_id).branches
        return [b.name for b in branches]
    except Exception as e:
        print(f"Error fetching branches: {e}")
        return []


##############################################################################################################
import signal
import psutil

def print_memory_usage():
    memory = psutil.Process().memory_info()
    print(f"RAM Usage: {memory.rss / (1024 * 1024 * 1024):.2f} GB")

def cleanup_handler(signum, frame):
    print("\nCleaning up cached data...")
    torch.cuda.empty_cache()
    print_memory_usage()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_handler)
##############################################################################################################


def main():
    ensure_hf_login()
    
    # Initialize new wandb run
    wandb.init(
        project="Edutech",
        config={
            'epochs': 100,
            'batch_size': 120,
            'learning_rate': 0.001,
            'fp16': True,
            'compile': True,
        }
    )
    branch_name = wandb.run.name
    print(f"Created new run with ID: {branch_name}")

    # Setup HuggingFace repository and branch
    repo = setup_hf_repo(branch_name)

    # Load model
    model = YOLO('yolo11l.pt')

    # Enable torch.compile if available
    if torch.__version__ >= "2" and wandb.config.compile:
        print("Compiling model with torch.compile()...")
        model.model = torch.compile(model.model)

    # Add WandB callback
    add_wandb_callback(model)

    # Training loop
    try:
        model.train(
            data='combined_dataset/data.yaml',
            epochs=50,
            imgsz=640,
            batch=128,
            workers=32,
            exist_ok=True,
            device='0',
            project=wandb.run.project,
            verbose=True,
            amp=True,
            cache="ram",
            save_period=1
        )
        
        # Save final model
        print("Training completed. Saving final model...")
        save_final_model(model, repo, branch_name)
        
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        wandb.finish()

if __name__ == "__main__":
    main()




# Run inference using the best model
results = model.predict(
    source=os.path.join('combined_dataset', 'images/val'),
    save=True,
    project='runs/predict',
    name='yolo11x_predict',
    exist_ok=True
)



import matplotlib.pyplot as plt
import glob
import os

def visualize_predictions(result_dir):
    image_paths = glob.glob(os.path.join(result_dir, '*.jpg'))
    num_images = min(4, len(image_paths))

    plt.figure(figsize=(15, 12))
    for i, image_path in enumerate(image_paths[:num_images]):
        image = plt.imread(image_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions('runs/predict/yolo11x_predict')
