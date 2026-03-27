"""
Configuration for the GNN+RL submodular cover optimization.

Problem:
    Given an N×N Bessel-kernel Toeplitz matrix J and constants sigma, epsilon,
    find a minimum binary vector w (W = diag(w)) satisfying:
        trace(J - J·W·(W·J·W + sigma²·I)⁻¹·W·J) ≤ epsilon
    i.e., minimise sum(w) = 1^T·w (fewest sensors).
"""

# ──────────────────────────────────────────────────────────────────────────────
# Problem parameters
# ──────────────────────────────────────────────────────────────────────────────
N = 20                  # dimension of the Toeplitz matrix J (number of positions)
SIGMA = 0.5             # noise standard deviation
EPSILON_FRAC = 0.20     # constraint: trace ≤ epsilon_frac × trace(J)

# Bessel (Matérn) kernel parameters
KERNEL_NU = 1.5         # Matérn smoothness (0.5, 1.5, 2.5 give closed forms)
KERNEL_LENGTH_SCALE = 3.0  # spatial length-scale

# ──────────────────────────────────────────────────────────────────────────────
# GNN architecture
# ──────────────────────────────────────────────────────────────────────────────
NODE_FEAT_DIM = 4       # input feature dimension per node
HIDDEN_DIM = 64         # hidden dimension in GNN layers
N_GNN_LAYERS = 3        # number of message-passing layers

# ──────────────────────────────────────────────────────────────────────────────
# Reinforcement-learning / training
# ──────────────────────────────────────────────────────────────────────────────
LR = 3e-4               # Adam learning rate
GAMMA = 0.99            # discount factor for returns
ENTROPY_COEF = 0.02     # entropy bonus coefficient (exploration)
VALUE_LOSS_COEF = 0.5   # coefficient for value-function loss

N_TRAIN_EPISODES = 2000 # total training episodes
EVAL_INTERVAL = 100     # evaluate every N episodes
PRINT_INTERVAL = 50     # print training stats every N episodes
SEED = 42               # random seed

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoint.pt"  # where to save the trained model
PLOT_PATH = "training_curve.png"   # training-curve plot
