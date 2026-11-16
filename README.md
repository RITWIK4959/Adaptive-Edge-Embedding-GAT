# Adaptive Edge-Embedding GAT

**Adaptive Edge-Embedding Graph Attention Network (AE-GAT)**

A research project for evaluating graph attention networks with static, learned, and adaptive edge representations on citation-network benchmark datasets.

The project compares:

- GAT
- GATv2
- GATv2 + Static Edge
- GATv2 + Learned Edge
- AE-GAT

on:

- Cora
- CiteSeer
- PubMed

---

## Overview

Graph Attention Networks (GATs) learn node representations by assigning attention weights to neighboring nodes.

This project investigates whether incorporating edge representations can improve graph attention.

The experimental pipeline compares progressively more expressive models:

```text
                         ┌── GAT
                         │
                         ├── GATv2
                         │
Input Graph ─────────────┼── GATv2 + Static Edge
                         │
                         ├── GATv2 + Learned Edge
                         │
                         └── AE-GAT
```

### Models

| Model | Description |
|---|---|
| GAT | Standard Graph Attention Network |
| GATv2 | Dynamic attention mechanism |
| GATv2 + Static Edge | GATv2 using deterministic edge attributes |
| GATv2 + Learned Edge | GATv2 using learned source-target edge embeddings |
| AE-GAT | Adaptive edge-embedding GAT with gated edge representations |

The goal is not only to compare accuracy, but also to evaluate:

- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Trainable parameters
- Training time
- Validation performance

---

# 1. Datasets

The project uses the standard citation-network benchmarks provided through PyTorch Geometric:

```text
Cora
CiteSeer
PubMed
```

The datasets are loaded using the PyTorch Geometric `Planetoid` dataset interface.

The standard public Planetoid split is used.

Typical split sizes are:

| Dataset | Nodes | Classes | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Cora | 2708 | 7 | 140 | 500 | 1000 |
| CiteSeer | 3327 | 6 | 120 | 500 | 1000 |
| PubMed | 19717 | 3 | 60 | 500 | 1000 |

The datasets are automatically downloaded when they are not already available.

---

# 2. Requirements

## Python

Recommended:

```text
Python 3.11
```

Python 3.12 or newer may also work, but Python 3.11 is recommended for reproducible experiments.

Check the installed version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

# 3. Required Libraries

The project requires:

```text
PyTorch
PyTorch Geometric
NumPy
Pandas
OpenPyXL
Scikit-learn
```

The project contains a `requirements.txt` file.

## requirements.txt

```text
torch>=2.6,<2.15
torch-geometric>=2.6
numpy>=1.26
pandas>=2.2
openpyxl>=3.1
scikit-learn>=1.5
```

---

# 4. Installation

## Step 1: Clone the repository

```bash
git clone https://github.com/RITWIK4959/Adaptive-Edge-Embedding-GAT.git
```

Move into the project:

```bash
cd Adaptive-Edge-Embedding-GAT
```

---

# 5. Create a Virtual Environment

## Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Verify:

```bash
python --version
```

---

# 6. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 7. Install PyTorch

## CPU

The project can run entirely on CPU.

For a CPU-only installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## NVIDIA GPU

If the system has an NVIDIA GPU, install the PyTorch build appropriate for the system's CUDA version.

Use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Then verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected for a working CUDA installation:

```text
True
```

If CUDA is unavailable, the project automatically runs on CPU.

---

# 8. Install Project Dependencies

After installing PyTorch:

```bash
pip install -r requirements.txt
```

Verify the important packages:

```bash
python -c "import torch; import torch_geometric; import numpy; import pandas; import sklearn; import openpyxl; print('All dependencies installed successfully')"
```

---

# 9. Dataset Setup

No manual dataset download is normally required.

Run:

```bash
python -m src.main
```

PyTorch Geometric downloads the required Planetoid datasets automatically if they are not already present.

The datasets are stored under:

```text
data/
```

After downloading, the project will contain directories similar to:

```text
data/
├── Cora/
├── CiteSeer/
└── PubMed/
```

Do not commit the downloaded datasets to Git unless explicitly required.

---

# 10. Project Structure

```text
Adaptive-Edge-Embedding-GAT/
│
├── data/
│   ├── Cora/
│   ├── CiteSeer/
│   └── PubMed/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── edge_encoder.py
│   ├── evaluate.py
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   └── tune.py
│
├── results/
│   ├── GAT_Research_Results.xlsx
│   ├── all_results.csv
│   └── all_results.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

The `results/` directory is generated after the experiment is executed.

---

# 11. Main Files

## `src/main.py`

Main entry point.

It:

- Loads Cora
- Loads CiteSeer
- Loads PubMed
- Creates all five models
- Trains the models
- Evaluates validation performance
- Restores the best validation checkpoint
- Evaluates the test set
- Generates result files

Run the project using:

```bash
python -m src.main
```

---

## `src/config.py`

Contains experiment configuration such as:

- Random seed
- Dataset names
- Device
- Hidden dimensions
- Number of attention heads
- Dropout
- Learning rate
- Weight decay
- Number of epochs
- Early stopping patience

---

## `src/data_loader.py`

Responsible for:

- Dataset loading
- Feature normalization
- Device placement
- Dataset statistics

---

## `src/models.py`

Contains the five graph attention models:

```text
GAT
GATv2
GATv2 + Static Edge
GATv2 + Learned Edge
AE-GAT
```

---

## `src/edge_encoder.py`

Implements edge representation methods:

```text
Static Edge Attributes
Learned Edge Encoder
Adaptive Edge Encoder
```

---

## `src/train.py`

Responsible for:

- Model training
- Validation
- Loss calculation
- Optimization
- Learning-rate scheduling
- Gradient clipping
- Early stopping
- Best-model checkpointing
- Test evaluation

---

## `src/evaluate.py`

Responsible for saving experiment results to:

```text
Excel
CSV
JSON
```

---

## `src/tune.py`

Contains validation-based hyperparameter tuning utilities.

The test set should not be used for hyperparameter selection.

---

# 12. Run the Project

From the project root:

```bash
python -m src.main
```

### Important

Run the command from:

```text
Adaptive-Edge-Embedding-GAT/
```

Correct:

```bash
cd Adaptive-Edge-Embedding-GAT
python -m src.main
```

Do not run:

```bash
cd src
python -m src.main
```

because the project uses Python package-relative imports.

---

# 13. Training and Testing

Training and testing are handled by the main experiment runner.

Use:

```bash
python -m src.main
```

The workflow is:

```text
Dataset
   │
   ▼
Model initialization
   │
   ▼
Training set
   │
   ▼
Model training
   │
   ▼
Validation evaluation
   │
   ▼
Best validation checkpoint
   │
   ▼
Restore best checkpoint
   │
   ▼
Test evaluation
   │
   ▼
Save results
```

The test set is evaluated after selecting the best validation checkpoint.

This prevents the test set from being used as a training/tuning signal.

---

# 14. Model Configuration

The final baseline configuration uses a compact GAT architecture.

Typical settings include:

```text
Hidden channels: 8
Attention heads: 8
Attention dropout: 0.6
Classifier dropout: 0.6
Learning rate: 0.005
Weight decay: 0.0005
Epochs: 200
Early stopping patience: 40
```

These settings are intentionally kept close to a conventional GAT benchmark configuration rather than using a large custom network.

---

# 15. Results

After a successful run, the following files are generated:

```text
results/
├── GAT_Research_Results.xlsx
├── all_results.csv
└── all_results.json
```

---

## Excel Results

```text
results/GAT_Research_Results.xlsx
```

The workbook contains:

```text
All Results
Cora
CiteSeer
PubMed
Model Average
Dataset Average
Best Per Dataset
```

---

## CSV Results

```text
results/all_results.csv
```

Contains the complete experiment results in tabular format.

---

## JSON Results

```text
results/all_results.json
```

Contains the same experiment results in JSON format.

---

# 16. Evaluation Metrics

The project reports:

### Accuracy

Percentage of correctly classified nodes.

### Macro Precision

Average precision across all classes, giving each class equal importance.

### Macro Recall

Average recall across all classes.

### Macro F1

Average F1 score across classes.

### Weighted F1

F1 score weighted according to class support.

### Trainable Parameters

Number of parameters updated during training.

### Training Time

Time required to train each model.

### Best Epoch

Epoch at which the best validation checkpoint was obtained.

---

# 17. Reproducibility

The project uses:

```text
Seed = 42
```

The seed is applied to:

- Python
- NumPy
- PyTorch
- CUDA when available

Run the same command:

```bash
python -m src.main
```

to reproduce the experiment.

Small numerical differences may occur across:

- Operating systems
- PyTorch versions
- PyTorch Geometric versions
- CPU architectures
- GPU architectures

---

# 18. CPU Execution

The project automatically detects the available hardware.

For a CPU system:

```text
Device : cpu
```

No additional configuration is required.

For example:

```bash
python -m src.main
```

will execute the complete experiment on CPU.

---

# 19. GPU Execution

If CUDA-enabled PyTorch is installed correctly:

```python
torch.cuda.is_available()
```

should return:

```text
True
```

The project then automatically uses:

```text
cuda
```

instead of:

```text
cpu
```

---

# 20. Expected Console Output

A successful run starts approximately as:

```text
======================================================================
ADAPTIVE EDGE-EMBEDDING GAT
======================================================================
Device  : cpu
Seed    : 42
Datasets: Cora, CiteSeer, PubMed
Models  : 5
Hidden  : 8
Heads   : 8
Dropout : 0.6
======================================================================
```

The program then runs:

```text
Cora | GAT
Cora | GATv2
Cora | GATv2 + Static Edge
Cora | GATv2 + Learned Edge
Cora | AE-GAT

CiteSeer | GAT
CiteSeer | GATv2
...

PubMed | GAT
PubMed | GATv2
...
```

At the end:

```text
FINAL RESEARCH METRICS
```

followed by the complete model comparison.

---

# 21. Troubleshooting

## `ModuleNotFoundError`

Activate the virtual environment:

```bash
source venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

---

## Check PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Check PyTorch Geometric

```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

---

## Check CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset Download Failure

If a dataset download is incomplete, remove the affected dataset directory and rerun:

```bash
python -m src.main
```

For example:

```bash
rm -rf data/Cora
```

Then:

```bash
python -m src.main
```

---

# 22. Clean Installation

A new Linux/macOS machine can be configured using:

```bash
git clone https://github.com/RITWIK4959/Adaptive-Edge-Embedding-GAT.git

cd Adaptive-Edge-Embedding-GAT

python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python -m src.main
```

For an NVIDIA GPU, install the appropriate PyTorch CUDA build instead of the CPU wheel.

---

# 23. Research Workflow

```text
                Citation Graph
                       │
                       ▼
                Node Features
                       │
                       ▼
              ┌────────────────┐
              │ Graph Attention│
              │    Models      │
              └────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
      GAT            GATv2        Edge-aware
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                           Static   Learned   Adaptive
                           Edge      Edge      Edge
                              │        │        │
                              └────────┼────────┘
                                       ▼
                                Node Classification
                                       │
                                       ▼
                              Validation Evaluation
                                       │
                                       ▼
                             Best Checkpoint
                                       │
                                       ▼
                                Test Evaluation
                                       │
                                       ▼
                             Research Results
```

---

# 24. Experimental Fairness

For a fair model comparison:

- Use the same dataset split.
- Use the same random seed.
- Use the same evaluation metrics.
- Use the same training procedure.
- Select checkpoints using validation performance.
- Do not tune hyperparameters using the test set.
- Report results for every model.
- Report both accuracy and Macro F1.
- Report model parameter counts.

The test set should only be used for final evaluation.

---

# 25. Research Models

## GAT

Standard Graph Attention Network baseline.

```text
Node Features
      ↓
Multi-head GAT
      ↓
GAT Output
      ↓
Classification
```

## GATv2

Uses the GATv2 attention mechanism.

```text
Node Features
      ↓
GATv2
      ↓
Classification
```

## GATv2 + Static Edge

Adds deterministic edge attributes to GATv2.

```text
Node Features ──┐
                ├── GATv2
Edge Features ──┘
                ↓
          Classification
```

## GATv2 + Learned Edge

Learns edge representations from source and target node representations.

```text
Source Node ──┐
              ├── Edge Encoder ── Edge Embedding
Target Node ──┘
                         │
                         ▼
                      GATv2
                         │
                         ▼
                   Classification
```

## AE-GAT

Uses an adaptive edge encoder with gated edge representations.

```text
Source Node ──┐
Target Node ──┼── Adaptive Edge Encoder
Difference ───┤
Interaction ──┘
                    │
                    ▼
              Adaptive Edge
                 Embedding
                    │
                    ▼
                  GATv2
                    │
                    ▼
              Node Classification
```

---

# 26. Output Directory

Generated experiment files are stored in:

```text
results/
```

Example:

```text
results/
├── GAT_Research_Results.xlsx
├── all_results.csv
└── all_results.json
```

The results directory can be deleted before a fresh experiment:

```bash
rm -rf results
```

Then rerun:

```bash
python -m src.main
```

---

# 27. Quick Start

If Python and Git are already installed:

```bash
git clone https://github.com/RITWIK4959/Adaptive-Edge-Embedding-GAT.git

cd Adaptive-Edge-Embedding-GAT

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python -m src.main
```

That's all that is required for the standard CPU setup.

The datasets are downloaded automatically on first execution.

---

# 28. Main Entry Point

The main file is:

```text
src/main.py
```

Run it from the project root:

```bash
python -m src.main
```

---

# 29. License

This repository is intended for academic and research use.

Please comply with the licenses and terms associated with PyTorch, PyTorch Geometric, and the benchmark datasets used by the project.

---

# 30. Acknowledgements

This project uses:

- PyTorch
- PyTorch Geometric
- Cora
- CiteSeer
- PubMed

The graph-learning implementation is built using PyTorch Geometric's GAT/GATv2 operators and dataset infrastructure.

---

## Repository

**GitHub:**

https://github.com/RITWIK4959/Adaptive-Edge-Embedding-GAT

**Main execution command:**

```bash
python -m src.main
```