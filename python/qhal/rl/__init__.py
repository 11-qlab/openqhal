"""RL-based compiler for neutral atom rearrangement."""
from .env import AtomArrangementEnv
from .policy import PolicyValueNet, RolloutBuffer
from .policy_transformer import TransformerPolicy, TransformerAdapter
from .reinforce import train, evaluate, save_model, load_model, TrainConfig
from .ppo import PPOTrainer, PPOConfig
from .compiler import (
    RLCompiler, Move,
    moves_to_program, moves_to_schedule,
)

__all__ = [
    "AtomArrangementEnv",
    "PolicyValueNet", "RolloutBuffer",
    "TransformerPolicy", "TransformerAdapter",
    "train", "evaluate", "save_model", "load_model", "TrainConfig",
    "PPOTrainer", "PPOConfig",
    "RLCompiler", "Move",
    "moves_to_program", "moves_to_schedule",
]

# SB3 gym wrapper + features
try:
    from .gym_wrapper import ShapedAtomGym, ShapedAtomEnv
    from .features import AtomGridCNN
    __all__ += ["ShapedAtomGym", "ShapedAtomEnv", "AtomGridCNN"]
except ImportError:
    pass


# Classical planner + pipeline
try:
    from .planner import ClassicalPlanner, Move
    from .pipeline import RearrangementPipeline, PipelineResult
    __all__ += ["ClassicalPlanner", "Move",
                "RearrangementPipeline", "PipelineResult"]
except ImportError:
    pass
