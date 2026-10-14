# Google Drive Desktop Setup

## Purpose

This project keeps source code, configuration, documentation, and the EMOTIC dataset on the local machine. Google Drive is used only for artifacts that must remain available across Colab sessions and local development:

- Colab notebooks;
- training checkpoints;
- exported trained models.

The Git repository itself is not stored in Google Drive. In particular, `.git`, `src`, `configs`, and `data` remain local.

## Directory mapping

Google Drive for desktop exposes the project at:

```text
G:\My Drive\Colab Notebooks\Contextual Emotion Recognition
```

The local repository is located at:

```text
C:\Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning
```

The following Windows directory junctions connect the repository to Drive:

| Local path | Google Drive target | Purpose |
|---|---|---|
| `notebooks` | `notebooks` | Colab notebooks |
| `outputs/checkpoints` | `checkpoints` | Resumable and best checkpoints |
| `models` | `models` | Final exported models |

Everything else remains local.

## Resulting structure

```text
Local repository/
├── data/                       # Local only
├── src/                        # Local and versioned by Git
├── configs/                    # Local and versioned by Git
├── notebooks/                  # Junction to Google Drive
├── outputs/
│   ├── checkpoints/            # Junction to Google Drive
│   ├── figures/                # Local only
│   ├── logs/                   # Local only
│   └── predictions/            # Local only
└── models/                     # Junction to Google Drive

Google Drive project/
├── notebooks/
├── checkpoints/
└── models/
```

## Synchronization behavior

When a notebook is saved in Colab:

```text
Colab saves the notebook
    ↓
Google Drive stores the new version
    ↓
Google Drive for desktop updates the G: drive
    ↓
The change appears at <local repository>\notebooks
```

The local path is not a second copy. It points to the same Drive-managed directory through a junction.

The same behavior applies to files saved under `outputs/checkpoints` and `models`.

## Colab paths

Use the following paths after mounting Google Drive in Colab:

```python
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition"
)

NOTEBOOK_ROOT = DRIVE_ROOT / "notebooks"
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints"
MODEL_ROOT = DRIVE_ROOT / "models"

for path in (NOTEBOOK_ROOT, CHECKPOINT_ROOT, MODEL_ROOT):
    path.mkdir(parents=True, exist_ok=True)
```

Recommended checkpoint names:

```text
checkpoints/<experiment_name>/<run_id>/last.pt
checkpoints/<experiment_name>/<run_id>/best.pt
```

Recommended model export layout:

```text
models/<model_name>/<version>/
├── model.pt
├── config.json
├── label_map.json
├── metrics.json
└── README.md
```

## Local training paths

Local training code can continue using repository-relative paths:

```python
from pathlib import Path

REPO_ROOT = Path.cwd()
CHECKPOINT_ROOT = REPO_ROOT / "outputs" / "checkpoints"
MODEL_ROOT = REPO_ROOT / "models"
```

The junctions redirect writes to Google Drive automatically.

## Git policy

The following directories are ignored by Git:

```gitignore
notebooks/
outputs/
models/
```

This prevents notebooks, checkpoints, and trained models from being committed or uploaded through Git LFS. Source code and project documentation continue to use normal Git version control.

## Verification

Run the following command from PowerShell:

```powershell
$repoRoot = "C:\Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning"

Get-Item `
  "$repoRoot\notebooks", `
  "$repoRoot\outputs\checkpoints", `
  "$repoRoot\models" |
Select-Object FullName, LinkType, Target
```

All three entries must report `LinkType` as `Junction` and display a target under `G:\My Drive`.

## Operating notes

- Google Drive for desktop must be running for streamed files to remain accessible.
- Mark the project folder as **Available offline** if checkpoints or models must remain accessible without a network connection.
- Wait for Drive synchronization to finish before shutting down the computer or disconnecting a Colab runtime.
- Do not edit the same notebook simultaneously in Colab and a local editor.
- Deleting a file through either the local junction or the Drive folder deletes the same Drive-managed file.
- Do not place `.git` or the entire repository inside Google Drive.

## Local backups created during setup

The original empty local directories are retained as:

```text
notebooks.local-backup-20261014
outputs/checkpoints.local-backup-20261014
```

They can be removed after the junctions have been verified. They contain no project files at the time of setup.
