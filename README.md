# ReLearn - Reinforcement Learning Library

A comprehensive header-only C++ reinforcement learning library implementing various RL algorithms organized by categories.

## Structure

The library is organized according to the GUIDE.md structure, with each major category of RL algorithms in its own namespace and folder:

### 1. Model-Free, Value-Based (`model_free_value_based/`)
- **Q-Learning** (`q_learning.hpp`) - Tabular Q-learning for discrete state/action spaces
- **Deep Q-Network (DQN)** (`dqn.hpp`) - Neural network approximation of Q-function
- **DQN Variants** (`dqn_variants.hpp`) - Double DQN and Dueling DQN implementations

### 2. Model-Free, Policy-Gradient (`model_free_policy_gradient/`)
- **REINFORCE** (`reinforce.hpp`) - Basic policy gradient algorithm
- **TRPO** (`trpo.hpp`) - Trust Region Policy Optimization
- **PPO** (`ppo.hpp`) - Proximal Policy Optimization

### 3. Model-Free, Actor-Critic (`model_free_actor_critic/`)
- **A2C/A3C** (`a2c_a3c.hpp`) - Advantage Actor-Critic variants
- **DDPG** (`ddpg.hpp`) - Deep Deterministic Policy Gradient
- **TD3** (`td3.hpp`) - Twin Delayed Deep Deterministic Policy Gradient
- **SAC** (`sac.hpp`) - Soft Actor-Critic

### 4. Model-Based RL (`model_based/`)
- **PILCO** (`pilco.hpp`) - Probabilistic Inference for Learning Control
- **MBPO** (`mbpo.hpp`) - Model-Based Policy Optimization
- **Dreamer & PETS** (`dreamer_pets.hpp`) - Advanced model-based methods

### 5. Imitation & Inverse RL (`imitation_inverse/`)
- **Behavioral Cloning** (`behavioral_cloning.hpp`) - Supervised imitation learning
- **DAgger** (`dagger.hpp`) - Dataset Aggregation
- **GAIL** (`gail.hpp`) - Generative Adversarial Imitation Learning

### 6. Hierarchical & Meta-RL (`hierarchical_meta/`)
- **Options & Feudal Networks** (`options_feudal.hpp`) - Hierarchical RL methods
- **MAML & RL²** (`maml_rl2.hpp`) - Meta-learning algorithms

### 7. Evolutionary & Black-Box Methods (`evolutionary_blackbox/`)
- **CMA-ES & NES** (`cmaes_nes.hpp`) - Evolution strategies

### Common Utilities (`common/`)
- **Base Classes** (`base.hpp`) - Environment interface, replay buffer, utilities

## Usage

```cpp
#include <relearn/relearn.hpp>

using namespace relearn;

// Create a Q-Learning agent
model_free_value_based::QLearning<StateType, ActionType> agent;

// Create a replay buffer
common::ReplayBuffer<StateType, ActionType> buffer(capacity);

// Use utility functions
auto returns = common::Utils::compute_returns(rewards);
```

## Building

The library is header-only, so you just need to include the main header:

```cpp
#include <relearn/relearn.hpp>
```

## Testing

Tests are located in the `test/` directory and use doctest framework:

- `test_model_free_value_based.cpp`
- `test_model_free_policy_gradient.cpp`
- `test_model_free_actor_critic.cpp`
- `test_model_based.cpp`
- `test_imitation_inverse.cpp`
- `test_hierarchical_meta.cpp`
- `test_evolutionary_blackbox.cpp`
- `test_common.cpp`

## Examples

See `examples/` directory for usage examples:
- `library_demo.cpp` - Demonstrates the complete library structure
- `main.cpp` - Basic usage example

## Implementation Status

This is the initial structure with algorithm interfaces defined. Each algorithm class provides:
- Constructor and basic interface
- Method signatures for main functionality
- Template support for different state/action types
- Inline implementation ready for header-only usage

The actual algorithm implementations will be added incrementally while maintaining the established structure.

## License

This library structure follows the patterns established in the reference implementation from `simple_lib_to_copy_from/`.
