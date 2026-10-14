# Worklog

## 2026-10-14

### Project initialization and organization

- Initialized the repository for **Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning**.
- Added the dependencies required for image processing, MATLAB annotation loading, PyTorch, visualization, configuration, and evaluation to `requirements.txt`.
- Configured `.gitignore` and `.gitattributes` for the repository.

### EMOTIC data preparation

- Prepared both versions of the EMOTIC dataset: **PAMI** and **CVPR**.
- Added the PAMI `Annotations.mat` file and the CVPR `CVPR17_Annotations.mat` file.
- Extracted and organized the EMOTIC image sources, including `framesdb`, `mscoco`, `ade20k`, and `emodb_small`.

### Dataset loader

- Implemented `src/datasets/emotic_dataset.py` as a PyTorch `Dataset`.
- Added support for the `train`, `val`, and `test` splits in the EMOTIC annotations.
- Added support for one or more image roots so images can be resolved across different sources.
- Converted every annotated person into an independent sample, including images that contain multiple people.
- Implemented bounding-box handling, coordinate clamping to image dimensions, and optional padding around target-person crops.
- Added both the full context image and the target-person crop to each returned sample.
- Mapped the fixed set of 26 EMOTIC emotion labels to multi-hot tensors.
- Added `union` and `majority` policies for aggregating categorical annotations.
- Aggregated VAD annotations and created masks for samples without VAD values.
- Added strict missing-image validation and an option to skip samples whose images are unavailable during data inspection.

### Data inspection notebook

#### Purpose of `01_Inspect.ipynb`

- Created `01_Inspect.ipynb` as the data-validation stage that must be completed before model development or training.
- Used the notebook to verify the real structure of the original EMOTIC annotations instead of assuming a preprocessed schema.
- Verified that the PAMI and CVPR annotation files expose the expected `train`, `val`, and `test` splits.
- Counted image records and person annotations in each split to confirm that one image may contain multiple annotated target people.
- Checked whether every annotation record can be resolved to an existing image from the available EMOTIC image sources.
- Inspected target-person bounding boxes, categorical emotion annotations, and continuous VAD annotations.
- Visualized a source image with every target-person bounding box drawn on it so annotation coordinates can be checked against the actual image.
- Kept this notebook focused on inspection and validation; it does not train a model or generate processed training data.

#### Data and tools used

- Loaded the PAMI `Annotations.mat` file and the CVPR `CVPR17_Annotations.mat` file.
- Resolved images across the `framesdb`, `mscoco`, `ade20k`, and `emodb_small` sources.
- Used `scipy.io.loadmat(..., simplify_cells=True)` to convert MATLAB annotation structures into Python dictionaries, lists, and NumPy values.
- Used `numpy` to normalize MATLAB arrays and parse bounding-box coordinates.
- Used `pathlib.Path` to construct relative image paths and search the configured image roots.
- Used `PIL.Image` to open source images in RGB format.
- Used `matplotlib` and `matplotlib.patches.Rectangle` to display images and draw target-person bounding boxes.
- Added helper functions for normalizing MATLAB values, loading available splits, counting people, constructing relative image paths, and resolving image files.
- Removed the dependency on importing `EmoticDataset`, allowing the notebook to inspect the raw EMOTIC files directly.
- Removed the previous import-error output and syntax-checked every code cell.

### Experimental workflow

- Converted `workflow.md` to a **notebook-first** workflow in which notebooks are the main entry points for training and evaluation.
- Defined the experimental sequence: data inspection, three visual baselines, direct VLM, structured-reasoning VLM, result analysis, and LoRA fine-tuning.
- Defined the three visual baselines: `person_only`, `context_only`, and `person_context`.
- Selected `BCEWithLogitsLoss` for the 26-class multi-label classification task.
- Selected validation mAP as the metric for choosing the best model.
- Designed the per-run output convention covering configuration, logs, metrics, predictions, figures, and checkpoints.
- Defined `last.pt` as the resumable training checkpoint and `best.pt` as the best validation checkpoint.
- Added guidance for resuming training and exporting a finalized model for inference.
