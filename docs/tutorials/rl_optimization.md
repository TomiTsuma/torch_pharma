# RL Optimization Tutorial

Optimize molecular properties using Reinforcement Learning.

---

## Overview

In this tutorial, you'll learn to:

1. Create a molecular editing environment
2. Train an RL agent (PPO) to optimize molecules
3. Generate optimized molecules for specific properties
4. Evaluate results

**Time to complete:** ~20 minutes

---

## Background

Reinforcement Learning for molecular optimization treats molecule design as a sequential decision process:

- **State**: Current molecular graph
- **Actions**: Add/remove atoms and bonds
- **Reward**: Chemical property score (QED, LogP)
- **Goal**: Maximize cumulative reward

---

## Step 1: Setup

```python
from torch_pharma.rl import MoleculeEnv, PPOAgent
from torch_pharma.training import Trainer
from torch_pharma.evaluation import ScoringFunction
from rdkit import Chem
```

---

## Step 2: Create Environment

```python
# Create molecular environment
env = MoleculeEnv(
    initial_smiles="C",           # Start with a single carbon
    max_steps=10,                 # Maximum 10 modifications
    target_property="qed",        # Optimize QED score
    atom_types=["C", "N", "O", "F"],  # Allowed atoms
    allow_removal=True            # Allow atom removal
)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
```

---

## Step 3: Initialize Agent

```python
# Initialize PPO agent
agent = PPOAgent(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99,
    clip_epsilon=0.2
)
```

---

## Step 4: Train

```python
# Initialize trainer
trainer = Trainer(max_steps=10000)

# Train agent
trainer.fit(agent, env)
```

---

## Step 5: Generate Optimized Molecules

```python
molecules = []
qed_scores = []

for _ in range(100):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    # Get final molecule
    mol = env.get_current_mol()
    if mol:
        molecules.append(mol)
        qed_scores.append(ScoringFunction.qed_score(mol))

print(f"Generated {len(molecules)} molecules")
print(f"Average QED: {sum(qed_scores)/len(qed_scores):.3f}")
print(f"Best QED: {max(qed_scores):.3f}")
```

---

## Step 6: Analyze Results

```python
import matplotlib.pyplot as plt

# Plot QED distribution
plt.hist(qed_scores, bins=20, edgecolor='black')
plt.xlabel('QED Score')
plt.ylabel('Count')
plt.title('Distribution of Generated Molecules')
plt.show()

# Print best molecules
best_idx = sorted(range(len(qed_scores)), key=lambda i: qed_scores[i], reverse=True)[:5]
for idx in best_idx:
    smiles = Chem.MolToSmiles(molecules[idx])
    print(f"SMILES: {smiles}, QED: {qed_scores[idx]:.3f}")
```

---

## Complete Code

```python
from torch_pharma.rl import MoleculeEnv, PPOAgent
from torch_pharma.training import Trainer
from torch_pharma.evaluation import ScoringFunction
from rdkit import Chem

# 1. Create environment
env = MoleculeEnv(
    initial_smiles="C",
    max_steps=10,
    target_property="qed",
    atom_types=["C", "N", "O", "F"]
)

# 2. Initialize agent
agent = PPOAgent(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=3e-4
)

# 3. Train
trainer = Trainer(max_steps=10000)
trainer.fit(agent, env)

# 4. Generate molecules
molecules = []
for _ in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, _, done, _ = env.step(action)
    molecules.append(env.get_current_mol())

# 5. Evaluate
qed_scores = [ScoringFunction.qed_score(m) for m in molecules]
print(f"Average QED: {sum(qed_scores)/len(qed_scores):.3f}")
```

---

## Multi-Objective Optimization

Optimize multiple properties simultaneously:

```python
from torch_pharma.rl import CompositeReward, QEDReward, LogPReward, SAReward

# Combine rewards
reward_fn = CompositeReward(
    rewards=[QEDReward(), LogPReward(target=2.0), SAReward()],
    weights=[1.0, 0.5, 0.3]
)

env = MoleculeEnv(
    initial_smiles="C",
    max_steps=10,
    reward_function=reward_fn  # Custom reward
)
```

---

## Tips

1. **Start simple**: Begin with QED, then add objectives
2. **Tune hyperparameters**: LR, gamma, clip_epsilon matter
3. **Use constrained action spaces**: Limit atoms/bonds for chemistry rules
4. **Post-process**: Filter by validity, uniqueness, novelty
