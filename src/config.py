import torch

SEED = 42

DATASET_ROOT = "./data"
DATASETS = ("Cora", "CiteSeer", "PubMed")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CPU_NUM_THREADS = 4

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

# Classic GAT-style hidden representation
HIDDEN_CHANNELS = 8

# Classic GAT uses multi-head attention in the hidden layer
NUM_HEADS = 8

# Number of GNN layers
NUM_LAYERS = 2

# Edge feature size
EDGE_DIM = 8

# ---------------------------------------------------------
# REGULARIZATION
# ---------------------------------------------------------

# Strong dropout is important for the tiny Planetoid
# training sets.
FEATURE_DROPOUT = 0.0
ATTENTION_DROPOUT = 0.60
CLASSIFIER_DROPOUT = 0.60
EDGE_DROPOUT = 0.10

# ---------------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------------

# Close to the classic GAT training recipe
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4

EPOCHS = 200
PATIENCE = 40

GRADIENT_CLIP = 1.0

# ---------------------------------------------------------
# LR SCHEDULER
# ---------------------------------------------------------

LR_SCHEDULER_PATIENCE = 15
LR_SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5

# ---------------------------------------------------------
# EDGE ENCODERS
# ---------------------------------------------------------

EDGE_ENCODER_HIDDEN = 16
AE_EDGE_CONTEXT_DROPOUT = 0.10

# ---------------------------------------------------------
# LOSS
# ---------------------------------------------------------

# Do NOT use label smoothing here.
#
# The classic benchmark recipe uses ordinary cross entropy,
# and our objective is to reproduce a strong baseline first.
LABEL_SMOOTHING = 0.0

# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

DETERMINISTIC = True