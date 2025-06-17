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

## 📋 Algorithm Implementation Status

### ✅ Fully Implemented
- **Q-Learning** - Production-ready with advanced features (experience replay, eligibility traces, double Q-learning, multiple exploration strategies, thread safety, model persistence)

### 🚧 In Development (Phase 1 Priority)
- **SARSA** - On-policy temporal difference learning
- **Expected SARSA** - Expected value updates, often outperforms Q-learning
- **R-Learning** - Average reward optimization for continuing tasks
- **Hysteretic Q-Learning** - Non-stationary environment adaptation

### 📝 Planned Implementations

#### Model-Free Value-Based
- **DQN** - Deep Q-Network for high-dimensional states
- **Double DQN** - Reduced overestimation bias
- **Dueling DQN** - Separate value and advantage streams
- **Rainbow DQN** - Combined DQN improvements

#### Multi-Agent Learning (Phase 2 Priority)
- **Independent Q-Learning** - Baseline multi-agent approach
- **Joint Action Learner** - Policy modeling for coordination
- **Sparse Cooperative Q-Learning** - Scalable factored coordination
- **MAUCE** - Multi-agent upper confidence exploration

#### Model-Based (Phase 3 Priority)
- **Dyna-Q** - Integration of learning and planning
- **Prioritized Sweeping** - Efficient model-based updates
- **MCTS** - Monte Carlo Tree Search planning

#### Model-Free Policy Gradient
- **REINFORCE** - Basic policy gradient with Monte Carlo
- **PPO** - Proximal Policy Optimization
- **TRPO** - Trust Region Policy Optimization

#### Model-Free Actor-Critic
- **A2C/A3C** - Advantage Actor-Critic (sync/async)
- **DDPG** - Deep Deterministic Policy Gradient
- **TD3** - Twin Delayed Deep Deterministic
- **SAC** - Soft Actor-Critic

#### Advanced Exploration (Phase 4 Priority)
- **Thompson Sampling** - Bayesian exploration strategy
- **Information Gain Exploration** - Active learning approach

#### Other Categories
- **Imitation Learning**: Behavioral Cloning, GAIL, DAgger
- **Hierarchical RL**: Options, Feudal Networks, MAML
- **Evolutionary Methods**: CMA-ES, NES, Genetic Algorithms

### 🎯 Current Focus
**Phase 1**: Completing core value-based algorithms (SARSA, Expected SARSA)
**Phase 2**: Multi-agent learning foundations
**Phase 3**: Model-based planning algorithms
**Phase 4**: Advanced exploration and policy methods

> **Note**: See `IMPLEMENTATION_STATUS.md` for detailed progress tracking and `tutorials/RL_ALGORITHMS_GUIDE.md` for comprehensive implementation plan.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add comprehensive tests
4. Ensure all tests pass (`make test`)
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by classical RL literature (Sutton & Barto)
- Modern RL techniques from recent research

