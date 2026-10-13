# Contextual Emotion Recognition Project
## Folder Structure and Training Workflow

This document summarizes the recommended project structure and the complete workflow for developing, training, and evaluating the contextual emotion recognition system.

---

# 1. Project Goal

The project aims to build an explainable contextual emotion recognition system using:

- Target person information
- Full-scene context
- Multi-label emotion classification
- Vision-Language Models
- Structured text-guided reasoning

The recommended development order is:

```text
Dataset
  ↓
Visual Baseline
  ↓
Direct VLM
  ↓
Structured Reasoning VLM
  ↓
Evaluation
  ↓
Fine-tuning
```

---

# 2. Recommended Folder Structure

```text
contextual-emotion-reasoning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── emotic/
│   ├── processed/
│   └── samples/
│
├── configs/
│   ├── baseline.yaml
│   ├── vlm.yaml
│   └── reasoning.yaml
│
├── notebooks/
│   ├── 01_inspect_emotic.ipynb
│   ├── 02_test_baseline.ipynb
│   ├── 03_test_vlm.ipynb
│   └── 04_analyze_results.ipynb
│
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── emotic_dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── person_encoder.py
│   │   ├── context_encoder.py
│   │   ├── fusion.py
│   │   └── classifier.py
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── direct_vlm.py
│   │   ├── structured_vlm.py
│   │   └── parser.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   └── faiss_index.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── emotion_metrics.py
│   │   └── explanation_metrics.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── logger.py
│   │   └── visualization.py
│   │
│   ├── train_baseline.py
│   ├── evaluate.py
│   └── run_vlm.py
│
├── experiments/
│   ├── person_only/
│   ├── context_only/
│   ├── person_context/
│   ├── vlm_direct/
│   └── vlm_structured/
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── logs/
│   └── figures/
│
└── paper/
    ├── main.tex
    ├── references.bib
    └── figures/
```

---

# 3. Folder Responsibilities

## `data/`

Contains the dataset.

```text
data/raw/
```

Store the original EMOTIC files here.

Do not modify the raw data.

```text
data/processed/
```

Store processed annotations, cached crops, normalized labels, or generated metadata.

```text
data/samples/
```

Store a small subset for debugging.

---

## `configs/`

Contains experiment configuration files.

Example:

```yaml
model:
  backbone: resnet50
  pretrained: true

training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.0001

data:
  image_size: 224
```

This helps reproduce experiments without changing the Python code.

---

## `notebooks/`

Use notebooks for:

- Data inspection
- Visualization
- Quick debugging
- Prompt testing
- Result analysis

Do not use notebooks as the primary training pipeline.

Recommended role:

```text
Notebook = explore + debug + visualize
Python script = actual training
```

---

## `src/datasets/`

Contains dataset loading and preprocessing logic.

Main file:

```text
src/datasets/emotic_dataset.py
```

It should return something similar to:

```python
{
    "image": image,
    "person_crop": person_crop,
    "bbox": bbox,
    "labels": labels,
    "vad": vad
}
```

---

## `src/models/`

Contains the visual baseline architecture.

Example pipeline:

```text
Person Crop ──→ Person Encoder ──┐
                                 ├─→ Fusion → Emotion Classifier
Full Image ───→ Context Encoder ─┘
```

Possible encoders:

- ResNet50
- ViT
- CLIP

---

## `src/reasoning/`

Contains VLM prompting and structured reasoning.

Example:

```text
Image
 ↓
Visual Cues
 ↓
Context Interpretation
 ↓
Situation
 ↓
Emotion
```

Example structured output:

```json
{
  "person_cues": [
    "smiling",
    "relaxed posture"
  ],
  "context_cues": [
    "people nearby are celebrating"
  ],
  "situation": "social celebration",
  "emotions": [
    "Happiness",
    "Engagement"
  ]
}
```

---

## `src/evaluation/`

Contains evaluation metrics.

Emotion metrics:

- mAP
- Macro F1
- Micro F1
- Per-class AP
- Precision
- Recall

Explanation metrics can later include:

- Visual grounding
- Hallucination rate
- Emotion consistency
- Target-person consistency

---

# 4. Development Workflow

The project should be implemented in the following order.

---

# Phase 1 — Dataset Preparation

Do not train anything yet.

Start by implementing:

```text
src/datasets/emotic_dataset.py
```

Then inspect the dataset using:

```text
notebooks/01_inspect_emotic.ipynb
```

Check:

- Image loading
- Bounding boxes
- Person crops
- Emotion labels
- VAD values
- Train/validation/test split
- Cases with multiple people

Minimum milestone:

```text
Image + Bounding Box
        ↓
Person Crop + Full Scene
        ↓
26 Emotion Labels + VAD
```

Only continue when this pipeline is correct.

---

# Phase 2 — Train the Visual Baseline

This is the first stage where model training is required.

Run three experiments:

```text
Experiment A: Person only

Experiment B: Context only

Experiment C: Person + Context
```

Recommended first model:

```text
Person Crop ──→ ResNet50 ──┐
                           ├─→ Feature Fusion → 26 logits
Full Image ───→ ResNet50 ──┘
```

Because EMOTIC is multi-label, use:

```python
torch.nn.BCEWithLogitsLoss()
```

Do not use standard softmax cross-entropy for the 26 emotion labels.

Training command should eventually look like:

```bash
python src/train_baseline.py --config configs/baseline.yaml
```

Outputs:

```text
outputs/checkpoints/
outputs/logs/
outputs/predictions/
```

---

# Phase 3 — Evaluate the Baseline

Run:

```bash
python src/evaluate.py --checkpoint outputs/checkpoints/best.pt
```

Record:

- mAP
- Macro F1
- Micro F1
- Per-class AP

Keep results for all three experiments:

```text
Person only
Context only
Person + Context
```

This gives you the first meaningful ablation study.

---

# Phase 4 — Direct VLM Baseline

At this stage, no training is required yet.

Use an existing VLM such as:

- Qwen2.5-VL
- LLaVA-OneVision
- InternVL

Run inference:

```text
Image
 ↓
VLM
 ↓
Emotion Labels
```

Example prompt:

```text
Identify the emotions experienced by the target person.
Return only labels from the EMOTIC emotion set.
```

Run using:

```bash
python src/run_vlm.py --mode direct
```

Save predictions to:

```text
outputs/predictions/vlm_direct.json
```

---

# Phase 5 — Structured Reasoning VLM

Still no fine-tuning required.

Prompt the VLM using an explicit reasoning structure:

```text
1. Describe visible cues of the target person.
2. Describe relevant contextual cues.
3. Infer the situation.
4. Predict the emotion labels.
```

Pipeline:

```text
Image
 ↓
Person Cues
 ↓
Context Cues
 ↓
Situation
 ↓
Emotion
```

Run:

```bash
python src/run_vlm.py --mode structured
```

Save:

```text
outputs/predictions/vlm_structured.json
```

---

# 5. First Main Experiment

You should now have five experiments:

| Experiment | Person | Context | VLM | Structured Reasoning |
|---|---:|---:|---:|---:|
| A | ✓ |  |  |  |
| B |  | ✓ |  |  |
| C | ✓ | ✓ |  |  |
| D | ✓ | ✓ | ✓ |  |
| E | ✓ | ✓ | ✓ | ✓ |

Main research question:

```text
Does structured text-guided reasoning improve contextual emotion recognition?
```

---

# 6. Fine-Tuning Stage

Do not fine-tune the VLM until the zero-shot and prompting baselines are complete.

Fine-tuning comes later.

Possible pipeline:

```text
EMOTIC Images
     ↓
Teacher VLM
     ↓
Structured Rationales
     ↓
Filtering
     ↓
Reasoning Dataset
     ↓
LoRA Fine-Tuning
     ↓
Fine-Tuned VLM
```

Training data could contain:

```json
{
  "image_id": "...",
  "person_bbox": [0, 0, 100, 200],
  "person_cues": [],
  "context_cues": [],
  "situation": "",
  "emotion_labels": []
}
```

---

# 7. Online Training Workflow

The recommended workflow is:

```text
VS Code
   ↓
GitHub
   ↓
Kaggle / Colab / GPU Server
   ↓
Train
   ↓
Save Checkpoint
   ↓
Evaluate
```

---

# 8. Local Development in VS Code

Write and test the source code locally.

Example:

```bash
git clone <your-repository>
cd contextual-emotion-reasoning
```

Create environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Do quick tests locally.

Avoid full model training if your local GPU is not suitable.

---

# 9. Push Code to GitHub

After coding:

```bash
git add .
git commit -m "Add EMOTIC baseline"
git push
```

Do not upload the whole EMOTIC dataset to GitHub.

Recommended `.gitignore`:

```gitignore
data/raw/
data/processed/
outputs/checkpoints/
outputs/logs/
__pycache__/
.venv/
```

---

# 10. Train on Kaggle

Create a Kaggle Notebook and enable GPU.

Then:

```bash
!git clone https://github.com/<username>/<repo>.git
%cd <repo>
!pip install -r requirements.txt
```

Attach EMOTIC as a Kaggle Dataset or upload it separately.

Then train:

```bash
!python src/train_baseline.py --config configs/baseline.yaml
```

Check GPU:

```bash
!nvidia-smi
```

---

# 11. Train on Google Colab

Clone repository:

```bash
!git clone https://github.com/<username>/<repo>.git
%cd <repo>
```

Install dependencies:

```bash
!pip install -r requirements.txt
```

Enable:

```text
Runtime
→ Change runtime type
→ GPU
```

Then run:

```bash
!python src/train_baseline.py --config configs/baseline.yaml
```

---

# 12. Train on a GPU Server

If using AWS, RunPod, Lambda Labs, or another GPU server:

```text
VS Code
  ↓
Remote SSH
  ↓
GPU Server
```

Then you can work almost exactly as if the remote machine were local.

Example:

```bash
git clone https://github.com/<username>/<repo>.git
cd contextual-emotion-reasoning
pip install -r requirements.txt
python src/train_baseline.py --config configs/baseline.yaml
```

For long-running training, consider:

```bash
tmux
```

or:

```bash
nohup python src/train_baseline.py --config configs/baseline.yaml > train.log 2>&1 &
```

---

# 13. Checkpoint Strategy

Always save:

```text
Last checkpoint
Best validation checkpoint
```

Example:

```text
outputs/checkpoints/
├── last.pt
└── best.pt
```

The best model should be selected using a validation metric such as:

```text
validation mAP
```

Do not select the final model using the test set.

---

# 14. Experiment Logging

For every experiment, save:

```text
Experiment name
Model
Backbone
Learning rate
Batch size
Epochs
Dataset split
Random seed
Validation metrics
Test metrics
Checkpoint path
```

Example experiment folder:

```text
experiments/person_context/
├── config.yaml
├── metrics.json
├── train.log
└── notes.md
```

This is important for reproducibility and paper writing.

---

# 15. Recommended Training Order

Use this exact order:

```text
1. Inspect EMOTIC
2. Build dataset loader
3. Verify preprocessing
4. Train person-only baseline
5. Train context-only baseline
6. Train person + context baseline
7. Evaluate all visual baselines
8. Run direct VLM
9. Run structured VLM
10. Compare results
11. Build rationale dataset
12. Fine-tune VLM with LoRA
13. Add retrieval
14. Evaluate explanation faithfulness
```

---

# 16. What Requires Training?

| Stage | Training Required? |
|---|---|
| EMOTIC preprocessing | No |
| Dataset visualization | No |
| Person-only baseline | Yes |
| Context-only baseline | Yes |
| Person + Context baseline | Yes |
| Direct VLM prompting | No |
| Structured VLM prompting | No |
| LoRA VLM fine-tuning | Yes |
| CLIP + FAISS retrieval | Usually no |
| Explanation evaluation | No |

---

# 17. Minimum Viable Research Pipeline

If time or GPU is limited, complete at least:

```text
EMOTIC
 ↓
Person + Context Baseline
 ↓
Direct VLM
 ↓
Structured VLM
 ↓
mAP / F1 Comparison
```

This is sufficient to test the core hypothesis before implementing more expensive stages.

---

# 18. Final Research Pipeline

The full version can become:

```text
EMOTIC Image
      │
      ├── Target Person Crop
      │
      └── Full Scene
              ↓
        Visual Encoders
              ↓
        Context Fusion
              ↓
       Structured Cues
              ↓
          Situation
              ↓
      Emotion Reasoning
         │         │
         ▼         ▼
     Emotions   Explanation
```

---

# 19. Immediate Next Step

Start with only these files:

```text
contextual-emotion-reasoning/
├── data/
├── notebooks/
│   └── 01_inspect_emotic.ipynb
├── src/
│   └── datasets/
│       └── emotic_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

Your first milestone is not model accuracy.

Your first milestone is:

```text
Load one EMOTIC image correctly
→ identify one target person
→ draw the correct bounding box
→ crop the target person
→ read the correct emotion labels
→ read VAD values
```

Once this works reliably, begin baseline training.
