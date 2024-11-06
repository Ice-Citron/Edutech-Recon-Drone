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

        model_state_dict = model.state_dict()
            
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
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

def find_latest_checkpoint(repo_path, branch_name):
    """Find the latest checkpoint in the repository"""
    checkpoint_dir = os.path.join(repo_path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
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
        return None, 0
    
    latest_epoch = max(epochs)
    return os.path.join(checkpoint_dir, f"epoch_{latest_epoch}", f"model_epoch_{latest_epoch}.pt"), latest_epoch


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
    # Ensure logged in to Hugging Face
    ensure_hf_login()

    # Ask user whether to continue existing run or start new one
    print("\nAvailable options:")
    print("1. Start new training run")
    print("2. Continue existing run")
    choice = input("Enter your choice (1/2): ")

    if choice == "2":
        branches = get_available_branches()
        if not branches:
            print("No existing branches found. Starting new run.")
            choice = "1"
        else:
            print("\nAvailable branches:")
            for i, branch in enumerate(branches, 1):
                print(f"{i}. {branch}")
            branch_idx = int(input("\nSelect branch number: ")) - 1
            branch_name = branches[branch_idx]
            
            # Initialize wandb with existing run ID
            wandb.init(
                project="Edutech",
                id=branch_name,  # Use branch name as the run ID
                resume="must",
                config={
                    'epochs': 100,
                    'batch_size': 120,
                    'learning_rate': 0.001,
                    'fp16': True,
                    'compile': True,
                }
            )

    if choice == "1":
        # First initialize wandb to get a new run ID
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
        # Use the wandb run ID for the branch name
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

    # Load checkpoint if continuing existing run
    start_epoch = 0
    if choice == "2":
        checkpoint_path, start_epoch = find_latest_checkpoint(HF_REPO, branch_name)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint['optimizer_state_dict']:
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, 100, 5):  # Train in 5-epoch intervals
        print(f"\nStarting training from epoch {epoch} to {epoch + 5}")
        model.train(
            data='combined_dataset/data.yaml',
            epochs=5,
            imgsz=640,
            batch=120,
            workers=32,
            exist_ok=True,
            device='0',
            project=wandb.run.project,
            verbose=True,
            amp=True,
            resume=True if epoch > start_epoch else False,
            cache="ram",
        )
        
        # Save checkpoint
        save_checkpoint(model, epoch + 5, repo, branch_name)
        print(f"Checkpoint saved for epoch {epoch + 5}")

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
