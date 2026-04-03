# Reinforcement Learning API

RL components for molecular optimization.

---

## Overview

The `rl` module provides reinforcement learning capabilities for de novo molecule design and optimization.

```python
from torch_pharma.rl import (
    MoleculeEnv,
    ProteinEnv,
    PPOAgent,
    DQNAgent,
    SACAgent,
    QEDReward,
    LogPReward,
)
```

---

## Environments

### MoleculeEnv

Molecular editing environment for RL.

```python
class MoleculeEnv:
    """
    Reinforcement learning environment for molecular optimization.
    
    The agent edits a molecular graph by adding/removing atoms and bonds
    to optimize a chemical property.
    
    State: Current molecular graph
    Actions: Add/remove atom/bond, terminate
    Reward: Chemical property score (QED, LogP, etc.)
    """
```

#### Constructor

```python
def __init__(
    self,
    initial_smiles: str = "C",
    max_steps: int = 10,
    target_property: str = "qed",
    reward_function: Optional[Callable] = None,
    atom_types: List[str] = ["C", "N", "O", "F"],
    allow_removal: bool = True,
    allow_no_modification: bool = True
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `initial_smiles` | `str` | Starting molecule SMILES |
| `max_steps` | `int` | Maximum episode length |
| `target_property` | `str` | Property to optimize (qed, logp, sa) |
| `reward_function` | `Optional[Callable]` | Custom reward function |
| `atom_types` | `List[str]` | Allowed atom types |
| `allow_removal` | `bool` | Allow atom/bond removal |
| `allow_no_modification` | `bool` | Allow "do nothing" action |

**Example:**

```python
from torch_pharma.rl import MoleculeEnv

env = MoleculeEnv(
    initial_smiles="C",
    max_steps=10,
    target_property="qed"
)

# Reset environment
state = env.reset()

# Take action
next_state, reward, done, info = env.step(action)
```

#### Methods

```python
def reset(self) -> Dict:
    """
    Reset environment to initial state.
    
    Returns:
        Initial state dictionary
    """

def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
    """
    Execute action in environment.
    
    Args:
        action: Action index
    
    Returns:
        - next_state: New molecular state
        - reward: Reward value
        - done: Whether episode ended
        - info: Additional information
    """
```

---

### ProteinEnv

Protein-ligand interaction environment.

```python
class ProteinEnv:
    """
    Environment for protein-ligand optimization.
    
    Optimizes ligand structure to improve binding affinity
    to a target protein.
    """
```

#### Constructor

```python
def __init__(
    self,
    protein_file: str,
    initial_ligand: str,
    max_steps: int = 20,
    docking_program: str = "vina"
)
```

---

## Agents

### PPOAgent

Proximal Policy Optimization agent.

```python
class PPOAgent(nn.Module):
    """
    Proximal Policy Optimization (Schulman et al., 2017).
    
    On-policy actor-critic algorithm with clipped objective.
    """
```

#### Constructor

```python
def __init__(
    self,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `observation_dim` | `int` | State space dimension |
| `action_dim` | `int` | Number of possible actions |
| `hidden_dim` | `int` | Hidden layer size |
| `lr` | `float` | Learning rate |
| `gamma` | `float` | Discount factor |
| `gae_lambda` | `float` | GAE lambda parameter |
| `clip_epsilon` | `float` | PPO clipping parameter |
| `value_coef` | `float` | Value loss coefficient |
| `entropy_coef` | `float` | Entropy bonus coefficient |

**Example:**

```python
from torch_pharma.rl import PPOAgent

agent = PPOAgent(
    observation_dim=256,
    action_dim=38,  # Number of possible molecular edits
    lr=3e-4
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done, log_prob, value)
    
    agent.update()
```

---

### DQNAgent

Deep Q-Network agent.

```python
class DQNAgent(nn.Module):
    """
    Deep Q-Network (Mnih et al., 2015).
    
    Off-policy Q-learning with experience replay.
    """
```

#### Constructor

```python
def __init__(
    self,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    buffer_size: int = 100000,
    batch_size: int = 64,
    target_update: int = 1000
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `epsilon_start` | `float` | Initial exploration rate |
| `epsilon_end` | `float` | Final exploration rate |
| `epsilon_decay` | `float` | Epsilon decay rate |
| `buffer_size` | `int` | Replay buffer size |
| `target_update` | `int` | Steps between target network updates |

---

### SACAgent

Soft Actor-Critic agent.

```python
class SACAgent(nn.Module):
    """
    Soft Actor-Critic (Haarnoja et al., 2018).
    
    Maximum entropy RL with automatic temperature tuning.
    """
```

---

## Reward Functions

### QEDReward

Quantitative Estimate of Drug-likeness reward.

```python
class QEDReward:
    """
    Reward based on QED score (Bickerton et al., 2012).
    
    QED combines molecular weight, LogP, H-bond donors/acceptors,
    PSA, rotatable bonds, and aromatic rings into a single score.
    """
    
    def __init__(self, weight: float = 1.0)
    
    def __call__(self, mol: Chem.Mol) -> float:
        """Calculate QED reward."""
```

---

### LogPReward

Lipophilicity reward.

```python
class LogPReward:
    """
    Reward based on partition coefficient (LogP).
    
    Measures lipophilicity - important for drug absorption.
    """
    
    def __init__(
        self,
        target: float = 2.0,
        tolerance: float = 0.5
    )
```

---

### SAReward

Synthetic accessibility reward.

```python
class SAReward:
    """
    Reward based on synthetic accessibility score.
    
    Penalizes molecules that are difficult to synthesize.
    """
```

---

### CompositeReward

Combine multiple reward functions.

```python
class CompositeReward:
    """
    Linear combination of multiple reward functions.
    """
    
    def __init__(
        self,
        rewards: List[Reward],
        weights: List[float]
    )
```

**Example:**

```python
from torch_pharma.rl import (
    CompositeReward,
    QEDReward,
    LogPReward,
    SAReward
)

reward_fn = CompositeReward(
    rewards=[QEDReward(), LogPReward(), SAReward()],
    weights=[1.0, 0.5, 0.3]
)

env = MoleculeEnv(reward_function=reward_fn)
```

---

## Buffers

### ReplayBuffer

Experience replay buffer.

```python
class ReplayBuffer:
    """
    Circular buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int)
    
    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> None
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of transitions."""
```

---

## Training with RL

```python
from torch_pharma.rl import MoleculeEnv, PPOAgent
from torch_pharma.training import Trainer

# Setup
env = MoleculeEnv(target_property="qed")
agent = PPOAgent(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# Train
trainer = Trainer(max_steps=10000)
trainer.fit(agent, env)

# Generate optimized molecules
molecules = []
for _ in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, _, done, _ = env.step(action)
    
    molecules.append(env.get_current_mol())
```
