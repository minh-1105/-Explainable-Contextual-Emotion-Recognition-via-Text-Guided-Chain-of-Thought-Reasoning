# Explainable Contextual Emotion Recognition

## Notebook-first training workflow

This document defines the project structure and development workflow for contextual emotion recognition with EMOTIC, visual baselines, Vision-Language Models (VLMs), and structured text-guided reasoning.

The project follows these principles:

```text
Notebook = primary place for configuration, training, quick evaluation, and result saving
src/     = reusable modules for datasets, models, metrics, prompts, and utilities
Drive    = persistent storage for data, checkpoints, models, logs, and results
/content = fast temporary runtime storage that is lost when Colab resets
```

Each experiment is launched from a notebook. Running `python src/train_*.py` is not required for the main training workflow. Python scripts remain optional for non-Colab execution or future automation.

---

## 1. Research objective

The system uses:

- the full-scene image;
- a crop of the target person;
- the target person's bounding box;
- the 26 multi-label EMOTIC emotion categories;
- Valence-Arousal-Dominance (VAD) values;
- Vision-Language Models;
- structured reasoning based on the target person and scene context.

The main research question is:

```text
Does structured text-guided reasoning improve contextual emotion
recognition and the quality of its explanations?
```

Recommended development order:

```text
Data inspection
    ↓
Visual baselines
    ↓
Direct VLM
    ↓
Structured-reasoning VLM
    ↓
Comparison and error analysis
    ↓
Fine-tuning / LoRA if needed
```

---

## 2. Source-code structure

Recommended repository structure:

```text
contextual-emotion-reasoning/
├── README.md
├── SETUP.md
├── workflow.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── vlm.yaml
│   └── reasoning.yaml
├── notebooks/
│   ├── 01_Inspect.ipynb
│   ├── 02_Train_Visual_Baseline.ipynb
│   ├── 03_Evaluate_Visual_Baseline.ipynb
│   ├── 04_Run_Direct_VLM.ipynb
│   ├── 05_Run_Structured_VLM.ipynb
│   ├── 06_Analyze_Results.ipynb
│   └── 07_Finetune_VLM_LoRA.ipynb
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── emotic_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── person_encoder.py
│   │   ├── context_encoder.py
│   │   ├── fusion.py
│   │   └── classifier.py
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── direct_vlm.py
│   │   ├── structured_vlm.py
│   │   └── parser.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── emotion_metrics.py
│   │   └── explanation_metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── logger.py
│       └── visualization.py
└── tests/
```

Do not commit the following directories to GitHub:

```text
data/raw/
outputs/
checkpoints/
models/
.venv/
__pycache__/
```

---

## 3. Google Drive structure

The project is currently located at:

```text
/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition
```

Recommended persistent-storage structure:

```text
Contextual Emotion Recognition/
├── data/
│   ├── raw/
│   │   └── raw/
│   │       └── emotic/
│   │           ├── CVPR/
│   │           └── PAMI/
│   │               ├── Annotations/
│   │               │   └── Annotations/
│   │               │       └── Annotations.mat
│   │               └── emotic/
│   │                   └── emotic/
│   ├── processed/
│   └── samples/
├── notebooks/
│   ├── 01_Inspect.ipynb
│   ├── 02_Train_Visual_Baseline.ipynb
│   ├── 03_Evaluate_Visual_Baseline.ipynb
│   ├── 04_Run_Direct_VLM.ipynb
│   ├── 05_Run_Structured_VLM.ipynb
│   ├── 06_Analyze_Results.ipynb
│   └── 07_Finetune_VLM_LoRA.ipynb
├── outputs/
│   ├── runs/
│   │   └── <experiment_name>/
│   │       └── <run_id>/
│   │           ├── checkpoints/
│   │           │   ├── last.pt
│   │           │   └── best.pt
│   │           ├── logs/
│   │           │   ├── history.csv
│   │           │   └── train.log
│   │           ├── metrics/
│   │           │   ├── val_metrics.json
│   │           │   └── test_metrics.json
│   │           ├── predictions/
│   │           │   ├── val_predictions.json
│   │           │   └── test_predictions.json
│   │           ├── figures/
│   │           ├── config.json
│   │           ├── environment.txt
│   │           └── run_summary.json
│   └── comparisons/
│       ├── baseline_comparison.csv
│       └── vlm_comparison.csv
└── models/
    └── <model_name>/
        └── <version>/
            ├── model.pt
            ├── config.json
            ├── label_map.json
            ├── metrics.json
            └── README.md
```

The two output areas serve different purposes:

- `outputs/runs/.../checkpoints/` stores resumable training state, including the optimizer, scheduler, epoch, and optional random state.
- `models/.../` stores selected models for inference, evaluation, or release. These exports do not need to retain the complete optimizer state.

Keep the current `data/raw/raw/emotic` layout to avoid moving the large dataset. If the dataset is later flattened to `data/raw/emotic`, only `ANNOTATION_PATH` and `IMAGE_ROOT` need to be changed in the configuration cell.

---

## 4. Shared Colab configuration cell

Every notebook should begin with the same path convention.

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition"
)

DATA_ROOT = DRIVE_ROOT / "data"
OUTPUT_ROOT = DRIVE_ROOT / "outputs"
MODEL_ROOT = DRIVE_ROOT / "models"

ANNOTATION_PATH = (
    DATA_ROOT
    / "raw/raw/emotic/PAMI/Annotations/Annotations/Annotations.mat"
)

IMAGE_ROOT = (
    DATA_ROOT
    / "raw/raw/emotic/PAMI/emotic/emotic"
)

for path in (
    DATA_ROOT / "processed",
    DATA_ROOT / "samples",
    OUTPUT_ROOT / "runs",
    OUTPUT_ROOT / "comparisons",
    MODEL_ROOT,
):
    path.mkdir(parents=True, exist_ok=True)

assert DRIVE_ROOT.is_dir(), f"Project not found: {DRIVE_ROOT}"
assert ANNOTATION_PATH.is_file(), f"Annotation file not found: {ANNOTATION_PATH}"
assert IMAGE_ROOT.is_dir(), f"Image root not found: {IMAGE_ROOT}"

print("Drive root:", DRIVE_ROOT)
print("Annotation:", ANNOTATION_PATH)
print("Image root:", IMAGE_ROOT)
```

Do not pass a Google Drive URL to `Path`. A URL such as `https://drive.google.com/...` is not a Colab filesystem path.

---

## 5. Preparing source code in the runtime

Notebooks are stored persistently, but source code should be cloned into `/content` for faster imports and file access.

```python
REPO_URL = (
    "https://github.com/minh-1105/"
    "-Explainable-Contextual-Emotion-Recognition-via-"
    "Text-Guided-Chain-of-Thought-Reasoning.git"
)
CODE_ROOT = Path("/content/contextual-emotion-reasoning")
```

Clone or update the repository:

```python
import subprocess

if not CODE_ROOT.exists():
    subprocess.run(
        ["git", "clone", REPO_URL, str(CODE_ROOT)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(CODE_ROOT), "pull", "--ff-only"],
        check=True,
    )
```

Install dependencies and add the repository to the Python import path:

```python
import sys
import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(CODE_ROOT / "requirements.txt")],
    check=True,
)

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
```

Repository modules can then be imported normally:

```python
from src.datasets.emotic_dataset import EmoticDataset
```

After a Colab runtime reset, rerun the Drive mount, repository clone, dependency installation, and path-configuration cells. Persistent data and checkpoints remain available.

---

## 6. Notebook responsibilities

### `01_Inspect.ipynb`

Objectives:

- inspect `Annotations.mat` and `CVPR17_Annotations.mat`;
- load the PAMI and CVPR `train`, `val`, and `test` splits;
- display the full image and target-person crop;
- draw and validate bounding boxes;
- inspect the 26 emotion categories and VAD values;
- inspect images with multiple annotated people;
- report missing images.

Do not start training until image paths, bounding boxes, person crops, labels, and VAD values have been verified.

### `02_Train_Visual_Baseline.ipynb`

This is the primary visual-baseline training notebook.

It must support three configurations:

```text
person_only
context_only
person_context
```

The notebook should:

1. mount persistent storage and prepare the source code;
2. define the experiment configuration;
3. create datasets and data loaders;
4. construct the model;
5. run the training and validation loops;
6. save `last.pt` after every completed epoch;
7. update `best.pt` when validation mAP improves;
8. write training history, metrics, and configuration to persistent storage;
9. support resuming from `last.pt`;
10. export the best model to `models/` after training finishes.

EMOTIC is a multi-label problem. The default loss is:

```python
criterion = torch.nn.BCEWithLogitsLoss()
```

Do not use softmax cross-entropy for the 26 emotion labels.

### `03_Evaluate_Visual_Baseline.ipynb`

This notebook loads `best.pt` and computes:

- mAP;
- Macro F1;
- Micro F1;
- per-class AP;
- precision and recall;
- multi-label error analysis;
- comparisons among `person_only`, `context_only`, and `person_context`.

### `04_Run_Direct_VLM.ipynb`

Run direct VLM prompting without fine-tuning:

```text
Image + target person
    ↓
VLM
    ↓
EMOTIC emotion labels
```

### `05_Run_Structured_VLM.ipynb`

Use a structured prompt:

```text
1. Observe visible cues from the target person.
2. Observe relevant contextual cues.
3. Infer the situation.
4. Predict the emotion labels.
5. Return the result with a fixed schema.
```

Example output:

```json
{
  "person_cues": ["smiling", "relaxed posture"],
  "context_cues": ["people nearby are celebrating"],
  "situation": "social celebration",
  "emotions": ["Happiness", "Engagement"]
}
```

### `06_Analyze_Results.ipynb`

Aggregate metrics, visualizations, successful and failed examples, and ablation results.

### `07_Finetune_VLM_LoRA.ipynb`

Run this notebook only after stable direct-VLM and structured-VLM zero-shot baselines are available.

---

## 7. Configuring a training run

Every run must have a unique identifier, and its configuration must be stored with its checkpoints.

```python
from datetime import datetime

TRAIN_CONFIG = {
    "experiment_name": "person_context",
    "backbone": "resnet50",
    "pretrained": True,
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 2,
    "epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "category_policy": "union",
    "crop_padding": 0.05,
    "seed": 42,
    "best_metric": "val_map",
}

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = (
    OUTPUT_ROOT
    / "runs"
    / TRAIN_CONFIG["experiment_name"]
    / RUN_ID
)

CHECKPOINT_DIR = RUN_DIR / "checkpoints"
LOG_DIR = RUN_DIR / "logs"
METRIC_DIR = RUN_DIR / "metrics"
PREDICTION_DIR = RUN_DIR / "predictions"
FIGURE_DIR = RUN_DIR / "figures"

for path in (
    CHECKPOINT_DIR,
    LOG_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    FIGURE_DIR,
):
    path.mkdir(parents=True, exist_ok=True)
```

Save the configuration immediately after creating the run:

```python
import json

with (RUN_DIR / "config.json").open("w", encoding="utf-8") as file:
    json.dump(TRAIN_CONFIG, file, ensure_ascii=False, indent=2)
```

Avoid fixed names such as `run1`, which make it easy to overwrite earlier results.

---

## 8. Checkpoint policy

Each training checkpoint should contain at least:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": (
        scheduler.state_dict() if scheduler is not None else None
    ),
    "best_val_map": best_val_map,
    "config": TRAIN_CONFIG,
    "class_names": list(train_dataset.class_names),
}
```

Conventions:

- `last.pt` is written after every completed epoch and is used to resume an interrupted run.
- `best.pt` is updated only when validation mAP improves.
- Never select the best model with the test set.
- Never overwrite a checkpoint belonging to a different run.

Persistent storage is slower than the runtime disk. Save a complete temporary checkpoint under `/content` and then copy it to persistent storage:

```python
import shutil
import torch

LOCAL_CHECKPOINT = Path("/content/checkpoint.pt")

torch.save(checkpoint, LOCAL_CHECKPOINT)
shutil.copy2(LOCAL_CHECKPOINT, CHECKPOINT_DIR / "last.pt")

if is_best:
    shutil.copy2(LOCAL_CHECKPOINT, CHECKPOINT_DIR / "best.pt")
```

This reduces the chance of leaving a partially written checkpoint if the runtime disconnects during `torch.save`.

---

## 9. Resuming training

Select the checkpoint from the intended run explicitly:

```python
RESUME_CHECKPOINT = (
    OUTPUT_ROOT
    / "runs/person_context/<run_id>/checkpoints/last.pt"
)
```

Restore the training state:

```python
checkpoint = torch.load(
    RESUME_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

start_epoch = checkpoint["epoch"] + 1
best_val_map = checkpoint["best_val_map"]
```

Before resuming, verify that the current configuration is compatible with the checkpoint, especially the backbone, number of classes, image size, and fusion architecture.

---

## 10. Exporting the trained model

A checkpoint is used to continue training; an exported model is used for inference. After selecting `best.pt`, create a versioned model directory:

```python
MODEL_NAME = "emotic_person_context_resnet50"
MODEL_VERSION = RUN_ID
EXPORT_DIR = MODEL_ROOT / MODEL_NAME / MODEL_VERSION
EXPORT_DIR.mkdir(parents=True, exist_ok=False)
```

Store the following files:

```text
model.pt       # best model_state_dict
config.json    # architecture and preprocessing configuration
label_map.json # fixed order of the 26 EMOTIC labels
metrics.json   # validation and test metrics
README.md      # source run, loading instructions, and model limitations
```

Example model-state export:

```python
best_checkpoint = torch.load(
    CHECKPOINT_DIR / "best.pt",
    map_location="cpu",
    weights_only=False,
)

torch.save(
    best_checkpoint["model_state_dict"],
    EXPORT_DIR / "model.pt",
)
```

The model-specific `README.md` should record at least:

- the experiment name and `run_id`;
- the Git commit used for training;
- the dataset split;
- preprocessing steps;
- backbone;
- validation and test metrics;
- training date;
- source notebook.

---

## 11. Logging and reproducibility

Every run must record:

- the complete configuration;
- random seed;
- Python, PyTorch, CUDA, and GPU versions;
- Git commit;
- loss per epoch;
- validation metrics per epoch;
- best checkpoint;
- final test metrics;
- start and end times.

Environment-recording cell:

```python
import platform
import subprocess
import torch

environment_lines = [
    f"python={platform.python_version()}",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
]

commit = subprocess.run(
    ["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

environment_lines.append(f"git_commit={commit}")
(RUN_DIR / "environment.txt").write_text(
    "\n".join(environment_lines) + "\n",
    encoding="utf-8",
)
```

---

## 12. Training order

Run experiments in this order:

```text
1. Run 01_Inspect.ipynb and verify the data.
2. Train person_only in 02_Train_Visual_Baseline.ipynb.
3. Train context_only.
4. Train person_context.
5. Evaluate all three baselines in 03_Evaluate_Visual_Baseline.ipynb.
6. Run the direct VLM in 04_Run_Direct_VLM.ipynb.
7. Run the structured VLM in 05_Run_Structured_VLM.ipynb.
8. Aggregate results in 06_Analyze_Results.ipynb.
9. Fine-tune with LoRA only after all baselines are complete.
```

Minimum experiment matrix:

| Experiment | Person | Context | VLM | Structured reasoning |
|---|---:|---:|---:|---:|
| A — Person only | ✓ |  |  |  |
| B — Context only |  | ✓ |  |  |
| C — Person + Context | ✓ | ✓ |  |  |
| D — Direct VLM | ✓ | ✓ | ✓ |  |
| E — Structured VLM | ✓ | ✓ | ✓ | ✓ |

---

## 13. Pre-training checklist

- [ ] Persistent storage is mounted correctly.
- [ ] `ANNOTATION_PATH.is_file()` returns `True`.
- [ ] `IMAGE_ROOT.is_dir()` returns `True`.
- [ ] Source code has been cloned into `/content`.
- [ ] Required dataset modules can be imported.
- [ ] Images, bounding boxes, and target-person crops are correct.
- [ ] The 26-label order is fixed.
- [ ] Train, validation, and test samples are not mixed.
- [ ] GPU acceleration is enabled.
- [ ] `RUN_DIR` points to a new directory.
- [ ] The configuration has been saved.
- [ ] `last.pt` and `best.pt` point to the intended run.
- [ ] Validation mAP is used to choose the best model.
- [ ] A newly saved checkpoint can be loaded successfully.

---

## 14. First milestone

The first milestone is not model accuracy.

```text
Load one EMOTIC image correctly
    ↓
Identify the correct target person
    ↓
Draw the correct bounding box
    ↓
Crop the correct person
    ↓
Read the correct emotion labels and VAD values
    ↓
Train on one batch
    ↓
Save and reload a checkpoint successfully
```

Do not begin long-running experiments until this complete sequence works reliably in the notebook.
