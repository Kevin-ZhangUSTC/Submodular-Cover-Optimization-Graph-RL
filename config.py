"""
Configuration for the GNN+RL submodular cover optimization.

Problem:
    Given an N×N kernel Toeplitz matrix J and constants sigma, epsilon,
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

# Kernel parameters
KERNEL_TYPE = "matern"  # "matern" or "j0"
KERNEL_NU = 1.5         # Matérn smoothness (0.5, 1.5, 2.5 give closed forms)
KERNEL_LENGTH_SCALE = 3.0  # spatial length-scale

# ──────────────────────────────────────────────────────────────────────────────
# GNN architecture
# ──────────────────────────────────────────────────────────────────────────────
NODE_FEAT_DIM = 8       # input feature dimension per node (8: 5 original + position + 2 Fourier)
HIDDEN_DIM = 64         # hidden dimension in GNN layers
N_GNN_LAYERS = 3        # number of message-passing layers
LAYER_TYPE = "sage"     # "sage" or "gat"
N_HEADS = 4             # attention heads (GAT only)
ATTENTION_DROPOUT = 0.0 # dropout on GAT attention weights
USE_RESIDUAL = False    # add residual connections to GNN layers
USE_ATTENTION_POOLING = False  # use attention pooling in value head
SIGNED_ADJ = False      # use signed adjacency split (for J0 kernel)

# ──────────────────────────────────────────────────────────────────────────────
# Scalability enhancements for large N (N=256) on J₀ kernels
# ──────────────────────────────────────────────────────────────────────────────
BAND_RADIUS = 0         # 0 = dense adj; >0 = sparse band-limited adj (O(N·band_radius))
STEP_PENALTY = 1.0      # per-step cost multiplier (-step_penalty/N per action); use 2.0
                        # for stronger parsimony pressure at large N
BEAM_WIDTH = 1          # number of stochastic rollouts at eval time; 1 = greedy decoding;
                        # >1 = beam search (run BEAM_WIDTH rollouts, take best result)
PERIOD_HINT = 0.0       # J₀ periodicity hint: set to 2.4 * length_scale to enable
                        # Fourier positional features (node features 6-7)

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
# Imitation learning (Idea 1)
# ──────────────────────────────────────────────────────────────────────────────
IMITATION_EPISODES = 200        # pure behavioural-cloning warm-start episodes
IMITATION_DECAY_EPISODES = 200  # episodes to anneal imitation_coef from 1 → 0
IMITATION_COEF = 1.0            # initial blended imitation coefficient

# ──────────────────────────────────────────────────────────────────────────────
# Multi-environment / generalisation training (Idea 4)
# ──────────────────────────────────────────────────────────────────────────────
MULTI_ENV = False       # enable multi-environment training
N_TRAIN_ENVS = 50       # number of training environments to generate
N_MIN = 5               # minimum N for generated environments
N_MAX = 30              # maximum N for generated environments
CURRICULUM = False      # enable curriculum scheduler

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoint.pt"  # where to save the trained model
PLOT_PATH = "training_curve.png"   # training-curve plot
