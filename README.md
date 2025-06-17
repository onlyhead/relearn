<img align="right" width="26%" src="./misc/logo.png">

ReLearn
==


Header-only C++ reinforcement learning library.

## ✨ Key Features

### 🎯 Q-Learning Implementation

The Q-learning implementation includes:

- **Multiple Exploration Strategies**: Epsilon-greedy, Boltzmann, UCB1, and epsilon decay
- **Advanced Learning Techniques**: Double Q-learning, eligibility traces, experience replay
- **Learning Rate Scheduling**: Constant, linear decay, exponential decay, and adaptive scheduling
- **Action Masking**: Environment-specific action filtering
- **Reward Shaping**: Custom reward transformation functions
- **Thread Safety**: Concurrent training with mutex protection
- **Performance Monitoring**: Comprehensive statistics tracking
- **Model Persistence**: Save/load Q-tables for deployment
- **Memory Management**: Efficient experience replay with capacity limits

### 🏗️ Architecture

```
relearn/
├── model_free_value_based/     # Q-Learning, DQN, Double DQN
├── model_free_policy_gradient/ # REINFORCE, PPO, TRPO
├── model_free_actor_critic/    # A2C/A3C, DDPG, TD3, SAC
├── model_based/               # PILCO, MBPO, Dreamer
├── imitation_inverse/         # Behavioral Cloning, GAIL, DAgger
├── hierarchical_meta/         # Options, Feudal Networks, MAML
├── evolutionary_blackbox/     # CMA-ES, NES
└── common/                    # Utilities, replay buffers
```

## 🛠️ Quick Start

### Basic Usage

```cpp
#include <relearn/relearn.hpp>
using namespace relearn::model_free_value_based;

// Create Q-learning agent
std::vector<int> actions = {0, 1, 2, 3};
QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

// Train the agent
agent.update(state, action, reward, next_state, terminal);
int best_action = agent.select_action(state);
```

### Advanced Configuration

```cpp
// Create agent with advanced features
QLearning<int, int> agent(
    0.1,  // learning rate
    0.9,  // discount factor
    0.1,  // exploration rate
    actions,
    QLearning<int, int>::ExplorationStrategy::BOLTZMANN,
    QLearning<int, int>::LearningRateSchedule::EXPONENTIAL_DECAY
);

// Enable advanced features
agent.set_double_q_learning(true);
agent.set_eligibility_traces(true, 0.9);
agent.set_experience_replay(true, 5000, 64);

// Set action masking
agent.set_action_mask([](int state, int action) {
    return is_valid_action(state, action);
});

// Set reward shaping
agent.set_reward_shaping([](double reward) {
    return reward * reward_scaling_factor;
});
```

## 📊 Performance

### Q-Learning Performance
- **Updates per second**: >100,000 updates/sec
- **Memory efficiency**: Sparse Q-table representation
- **Thread safety**: Full concurrent access support

### Test Coverage
- **Unit Tests**: 44 test assertions
- **Integration Tests**: Multi-threaded training validation
- **Performance Tests**: Benchmarking and memory efficiency
- **Feature Tests**: All advanced features validated

## 🔧 Advanced Features

### 1. Multiple Exploration Strategies

#### Epsilon-Greedy
```cpp
agent.set_exploration_strategy(ExplorationStrategy::EPSILON_GREEDY);
agent.set_epsilon(0.1);
```

#### Boltzmann Exploration
```cpp
agent.set_exploration_strategy(ExplorationStrategy::BOLTZMANN);
agent.set_temperature(2.0);
```

#### UCB1 (Upper Confidence Bound)
```cpp
agent.set_exploration_strategy(ExplorationStrategy::UCB1);
agent.set_ucb_c(1.4);
```

### 2. Double Q-Learning
Reduces overestimation bias using two Q-networks:
```cpp
agent.set_double_q_learning(true);
```

### 3. Eligibility Traces
Enables credit assignment for delayed rewards:
```cpp
agent.set_eligibility_traces(true, 0.9); // λ = 0.9
```

### 4. Experience Replay
Improves sample efficiency by replaying past experiences:
```cpp
agent.set_experience_replay(true, 10000, 64); // capacity=10k, batch=64
```

### 5. Learning Rate Scheduling
```cpp
// Exponential decay
agent.set_lr_schedule(LearningRateSchedule::EXPONENTIAL_DECAY);

// Linear decay
agent.set_lr_schedule(LearningRateSchedule::LINEAR_DECAY);

// Adaptive based on performance
agent.set_lr_schedule(LearningRateSchedule::ADAPTIVE);
```

### 6. Action Masking
Environment-specific action filtering:
```cpp
agent.set_action_mask([](int state, int action) {
    // Custom logic to determine valid actions
    return GridWorld::is_action_valid(state, action);
});
```

### 7. Reward Shaping
Transform rewards for better learning:
```cpp
agent.set_reward_shaping([](double reward) {
    return std::tanh(reward / 10.0); // Normalize large rewards
});
```

### 8. Model Persistence
Save and load trained models:
```cpp
// Save Q-table
agent.save_q_table("model.bin");

// Load Q-table
QLearning<int, int> new_agent(0.1, 0.9, 0.0, actions);
new_agent.load_q_table("model.bin");
```

### 9. Performance Monitoring
Track comprehensive statistics:
```cpp
auto stats = agent.get_statistics();
std::cout << "Total updates: " << stats.total_updates << std::endl;
std::cout << "Cumulative reward: " << stats.cumulative_reward << std::endl;
std::cout << "Exploration ratio: " << stats.exploration_ratio << std::endl;
std::cout << "Training time: " << stats.total_training_time.count() << " ms" << std::endl;
```

### 10. Thread Safety
Full support for concurrent training:
```cpp
QLearning<int, int> shared_agent(0.1, 0.9, 0.1, actions);

// Multiple threads can safely access the agent
std::vector<std::thread> workers;
for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back([&shared_agent]() {
        // Safe concurrent training
        shared_agent.update(state, action, reward, next_state, terminal);
        int action = shared_agent.select_action(state);
    });
}
```

## 🧪 Testing & Validation

### Test Suite
The library includes test coverage:

1. **Basic Functionality Tests** (22 assertions)
   - Constructor and parameter validation
   - Q-value operations
   - Action selection logic

2. **Advanced Feature Tests** (13 test cases)
   - All exploration strategies
   - Double Q-learning validation
   - Eligibility traces verification
   - Experience replay functionality
   - Learning rate scheduling
   - Action masking and reward shaping
   - Model persistence
   - Thread safety validation
   - Performance benchmarks

### Running Tests
```bash
cd build
make test
```

### Performance Benchmarks
```bash
./test_advanced_q_learning_comprehensive
```

## 🚀 Deployment

### Requirements
- **C++20** compatible compiler
- **CMake 3.15+** for building
- **Header-only**: No external dependencies

### Integration
```cpp
// Single include for full library
#include <relearn/relearn.hpp>

// Use specific namespaces
using namespace relearn::model_free_value_based;
using namespace relearn::common;
```

## 📈 Use Cases

### 1. Game AI
```cpp
// Chess/Go AI with action masking for legal moves
agent.set_action_mask([&game](int state, int action) {
    return game.is_legal_move(state, action);
});
```

### 2. Robotics Control
```cpp
// Robot navigation with reward shaping for smoother paths
agent.set_reward_shaping([](double reward) {
    return reward + smooth_path_bonus;
});
```

### 3. Financial Trading
```cpp
// Trading agent with experience replay for better sample efficiency
agent.set_experience_replay(true, 50000, 128);
agent.set_double_q_learning(true); // Reduce overestimation
```

### 4. Resource Management
```cpp
// Cloud resource allocation with UCB exploration
agent.set_exploration_strategy(ExplorationStrategy::UCB1);
agent.set_ucb_c(2.0);
```


## Overview
This document tracks the current implementation status of all algorithms in the ReLearn library, distinguishing between fully implemented, partially implemented, and placeholder/planned algorithms.

**Last Updated**: June 17, 2025

## Legend
- ✅ **Fully Implemented**: Complete implementation with tests and documentation
- 🚧 **Partially Implemented**: Basic structure exists but missing key features
- 📝 **Placeholder**: Empty/skeleton file with TODO comments
- ❌ **Not Started**: No file exists yet
- 🎯 **Priority**: Marked for immediate implementation

---

## Current Implementation Status

### Model-Free Value-Based Methods

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **Q-Learning** | ✅ **Fully Implemented** | `q_learning.hpp` | 730 | Experience replay, eligibility traces, double Q-learning, exploration strategies, thread safety, persistence | Complete |
| **SARSA** | 📝 **Placeholder** | `sarsa.hpp` | 50 | TODO: On-policy learning, eligibility traces | 🎯 **Phase 1** |
| **Expected SARSA** | 📝 **Placeholder** | `expected_sarsa.hpp` | 50 | TODO: Expected value updates | 🎯 **Phase 1** |
| **R-Learning** | 📝 **Placeholder** | `advanced_algorithms.hpp` | 50 | TODO: Average reward learning | Phase 1 |
| **Hysteretic Q-Learning** | 📝 **Placeholder** | `advanced_algorithms.hpp` | 50 | TODO: Different learning rates for +/- errors | Phase 1 |
| **DQN** | 📝 **Placeholder** | `dqn.hpp` | 44 | TODO: Neural network integration | Later |
| **Double DQN** | 📝 **Placeholder** | `dqn_variants.hpp` | - | TODO: Overestimation bias reduction | Later |
| **Dueling DQN** | ❌ **Not Started** | - | - | TODO: Value/advantage decomposition | Later |
| **Rainbow DQN** | ❌ **Not Started** | - | - | TODO: Combined improvements | Later |

### Model-Free Policy Gradient Methods

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **REINFORCE** | 📝 **Placeholder** | `reinforce.hpp` | - | TODO: Basic policy gradient | Phase 4 |
| **PPO** | 📝 **Placeholder** | `ppo.hpp` | 36 | TODO: Clipped surrogate objective | Phase 4 |
| **TRPO** | 📝 **Placeholder** | `trpo.hpp` | - | TODO: Trust region optimization | Phase 4 |
| **A2C** | 📝 **Placeholder** | `a2c_a3c.hpp` | - | TODO: Advantage actor-critic | Phase 4 |
| **A3C** | 📝 **Placeholder** | `a2c_a3c.hpp` | - | TODO: Asynchronous version | Phase 4 |

### Model-Free Actor-Critic Methods

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **DDPG** | 📝 **Placeholder** | `ddpg.hpp` | - | TODO: Continuous control | Phase 4 |
| **TD3** | 📝 **Placeholder** | `td3.hpp` | - | TODO: Twin delayed DDPG | Phase 4 |
| **SAC** | 📝 **Placeholder** | `sac.hpp` | - | TODO: Soft actor-critic | Phase 4 |
| **IMPALA** | ❌ **Not Started** | - | - | TODO: Distributed learning | Later |

### Model-Based Methods

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **Dyna-Q** | 📝 **Placeholder** | `planning_algorithms.hpp` | 100 | TODO: Model learning + planning | 🎯 **Phase 3** |
| **Prioritized Sweeping** | 📝 **Placeholder** | `planning_algorithms.hpp` | 100 | TODO: Priority-based updates | 🎯 **Phase 3** |
| **MCTS** | 📝 **Placeholder** | `planning_algorithms.hpp` | 100 | TODO: Tree search planning | 🎯 **Phase 3** |
| **PILCO** | 📝 **Placeholder** | `pilco.hpp` | - | TODO: Gaussian process models | Later |
| **MBPO** | 📝 **Placeholder** | `mbpo.hpp` | - | TODO: Model-based policy optimization | Later |
| **Dreamer** | 📝 **Placeholder** | `dreamer_pets.hpp` | - | TODO: Latent imagination | Later |
| **PlaNet** | ❌ **Not Started** | - | - | TODO: Planning network | Later |

### Multi-Agent Reinforcement Learning

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **Independent Q-Learning** | 📝 **Placeholder** | `joint_learning.hpp` | 73 | TODO: Parallel single-agent learning | 🎯 **Phase 2** |
| **Joint Action Learner** | 📝 **Placeholder** | `joint_learning.hpp` | 73 | TODO: Policy modeling | 🎯 **Phase 2** |
| **Sparse Cooperative Q-Learning** | 📝 **Placeholder** | `cooperative_learning.hpp` | 100 | TODO: Factored coordination | 🎯 **Phase 2** |
| **MAUCE** | 📝 **Placeholder** | `cooperative_learning.hpp` | 100 | TODO: Multi-agent UCB | Phase 2 |
| **Coordination Graph** | 📝 **Placeholder** | `cooperative_learning.hpp` | 100 | TODO: Graph-based coordination | Phase 2 |

### Imitation & Inverse Reinforcement Learning

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **Behavioral Cloning** | 📝 **Placeholder** | `behavioral_cloning.hpp` | - | TODO: Supervised imitation | Later |
| **GAIL** | 📝 **Placeholder** | `gail.hpp` | - | TODO: Adversarial imitation | Later |
| **DAgger** | 📝 **Placeholder** | `dagger.hpp` | - | TODO: Dataset aggregation | Later |
| **ValueDICE** | ❌ **Not Started** | - | - | TODO: Inverse RL | Later |

### Hierarchical & Meta-Learning

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **Options** | 📝 **Placeholder** | `options_feudal.hpp` | - | TODO: Temporal abstraction | Later |
| **Feudal Networks** | 📝 **Placeholder** | `options_feudal.hpp` | - | TODO: Hierarchical control | Later |
| **MAML** | 📝 **Placeholder** | `maml_rl2.hpp` | - | TODO: Meta-learning | Later |
| **Reptile** | ❌ **Not Started** | - | - | TODO: First-order meta-learning | Later |

### Evolutionary & Black-Box Methods

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **CMA-ES** | 📝 **Placeholder** | `cmaes_nes.hpp` | - | TODO: Evolution strategy | Later |
| **NES** | 📝 **Placeholder** | `cmaes_nes.hpp` | - | TODO: Natural evolution | Later |
| **OpenAI-ES** | ❌ **Not Started** | - | - | TODO: Distributed evolution | Later |
| **Genetic Algorithms** | ❌ **Not Started** | - | -| TODO: Population-based optimization | Later |

### Advanced Exploration Strategies

| Algorithm | Status | File | Lines | Features | Priority |
|-----------|---------|------|-------|----------|----------|
| **Thompson Sampling** | 📝 **Placeholder** | `advanced_exploration.hpp` | 50 | TODO: Bayesian exploration | 🎯 **Phase 4** |
| **Information Gain** | 📝 **Placeholder** | `advanced_exploration.hpp` | 50 | TODO: Active learning | Phase 4 |

---

## Multi-Agent Demo Status

### Working Implementations
- ✅ **Multi-Agent Harvest Demo**: Complete working example with 3 harvesting machines + 1 chaser
  - File: `examples/multi_agent_harvest_demo.cpp` (1142 lines)
  - Features: Territorial coordination, density-based harvesting, advanced Q-learning integration
  - Status: Fully functional, demonstrates multi-agent coordination

### Demo Applications
- ✅ **Q-Learning Demo**: Basic single-agent example
- ✅ **Library Demo**: Feature showcase
- 📝 **Multi-Agent Framework**: Uses existing single-agent Q-learning for coordination

---

## Current Architecture Analysis

### What's Actually Working
1. **Production-Ready Q-Learning**: Full implementation with all advanced features
2. **Multi-Agent Application**: Working harvest demo showing practical coordination
3. **Build System**: CMake build with tests and examples
4. **Header-Only Design**: Clean template-based architecture

### What Needs Implementation

#### Phase 1 Priority (🎯 Immediate)
1. **SARSA**: Complete on-policy learning algorithm
2. **Expected SARSA**: Bridge between Q-learning and SARSA
3. **Advanced Q-learning variants**: R-Learning, Hysteretic Q-Learning

#### Phase 2 Priority (🎯 Multi-Agent)
1. **Independent Q-Learning**: Multi-agent baseline
2. **Joint Action Learner**: Policy modeling for coordination
3. **Sparse Cooperative Q-Learning**: Scalable multi-agent coordination

#### Phase 3 Priority (🎯 Model-Based)
1. **Dyna-Q**: Model learning + planning integration
2. **Prioritized Sweeping**: Efficient model-based updates
3. **MCTS**: Tree search planning

#### Phase 4 Priority (Advanced)
1. **Thompson Sampling**: Advanced exploration
2. **Policy Gradient Methods**: REINFORCE, PPO
3. **Actor-Critic Methods**: A2C, DDPG, SAC
