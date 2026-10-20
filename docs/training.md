# Training

How the Recon Drone detector was trained, and how to train it again.

## Dataset

`training/prepare_dataset.py` downloads seven public Roboflow datasets in YOLO format. It merges them into `combined_dataset/` and maps the source classes onto eight target classes.

| Class | Source classes folded in |
|---|---|
| Glass | glass, glass bottle |
| Metal | metal, can, aluminium foil, pop tab |
| Plastic | plastic, PET bottle, plastic bag, plastic buoy, plastic container, bottle |
| Styrofoam | styrofoam box, buoy, piece, cup |
| Paper | paper, cardboard, carton |
| Rope | rope, net |
| Food Waste | food, decomposable |
| Other | other litter, straw, fabrics, wood, cigarette, wrapper, lid, bottle cap |

Sources on Roboflow Universe:

- `hongmo/sea-ezx3q` v44
- `kim-borma/marine-debris-vb2qm` v16
- `myfyp/new-taco` v3
- `anas-fd5bu/-kg` v5
- `trash-detection-eka7g/aeriel-trash-detection` v5
- `fyp-bfx3h/yolov8-trash-detections` v6 and `taco-ihjgk/yolov8-trash-detections-kgnug` v11

## Runs

| Run | Date | Model | Classes | Epochs | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| likely-fog-13 | 2024-10-21 | YOLOv8n | 11 | 3 | 0.578 | 0.413 |
| ethereal-sound-16 | 2024-10-21 | YOLO11x | 40 | 3 of 5, crashed | 0.418 | 0.315 |
| sleek-dream-43 | 2024-10-23 | YOLO11l | 8 | 5 | 0.324 | 0.218 |
| apricot-elevator-45 | 2024-10-23 | YOLO11l | 8 | 1 | 0.064 | 0.033 |
| **fluent-morning-79** | 2024-11-07 | YOLO11l | 8 | 9 | **0.381** | **0.264** |

Each number is on the validation split of that run's own dataset. Only the last three runs use the eight classes above. fluent-morning-79 trained with a script that had no W&B callback. Its metrics come from the `train_results` stored in its checkpoint.

<p align="center"><img src="wandb/plots/fluent-morning-79/results.png" width="900" alt="fluent-morning-79: validation metrics and losses per epoch"></p>

Charts and data:

- `wandb/panels/`: the 32 panels of the W&B report.
- `wandb/plots/<run>/`: the Ultralytics plots of each run (PR, F1, P and R curves, confusion matrices, prediction grids).
- `wandb/history/`: per-epoch metrics of every run as CSV.

The [W&B report](https://forge.coreweave.com/wandb/marlborough-college-malaysia/Edutech/reports/Recon-Drone-MCM-Edutech-Asia--VmlldzoxODE0OTE0OA) is interactive. [recon-drone-training-report.pdf](recon-drone-training-report.pdf) is the printed version.

## Settings of the final run

| Setting | Value |
|---|---|
| Model | `yolo11l.pt`, pretrained, 25.3 M parameters |
| Image size, batch | 640 px, 128 |
| Precision | AMP (fp16), `cache=ram`, 32 workers |
| Optimiser | Ultralytics auto, lr0 0.01, warm-up 3 epochs |
| Checkpoints | one per epoch, pushed to the Hugging Face repo |

The full argument list is in `training/configs/final_run_args.yaml`.

## Reproduce

```bash
export ROBOFLOW_API_KEY=...        # your own Roboflow key
cd training
python prepare_dataset.py          # downloads, merges and maps the datasets into combined_dataset/
wandb login && hf auth login
git clone https://huggingface.co/shng2025/EduTech-YOLOv11   # train.py writes checkpoints into this clone
python train.py                    # train_ddp.py is the multi-GPU variant
python inference.py                # predictions on combined_dataset/images/valid
```

## Notebooks

`training/notebooks/` holds the Colab notebooks in date order. They record the path from a ViT classifier in September 2024 to the YOLO11 checkpoint loop in October 2024. `docs/media/2024-10-04-yolov8-seg-demo.avi` is the output of the first YOLOv8 notebook.
