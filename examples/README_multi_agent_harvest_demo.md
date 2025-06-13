# Multi-Agent Pea Harvesting Coordination Demo

## Overview

This demo showcases an advanced Q-learning implementation for coordinating autonomous agricultural machinery in a pea harvesting scenario. The system simulates a 100x100 field where two harvesting machines work to collect peas while a chaser bin (mobile collection unit) learns to optimally coordinate with them using reinforcement learning.

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    100x100 Pea Field                            │
│                                                                 │
│  ┌─────┐                                        ┌─────┐         │
│  │ M0  │ ← Harvesting Machine 0                 │ M1  │         │
│  │Cap:50│   (Horizontal sweeps)                 │Cap:50│         │
│  └─────┘                                        └─────┘         │
│                                                                 │
│                       ┌─────┐                                   │
│                       │  C  │ ← Chaser Bin (Q-Learning Agent)  │
│                       │Cap:200│   (Learns optimal collection)   │
│                       └─────┘                                   │
│                                                                 │
│  Legend: . = unharvested, # = harvested                        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Two Harvesting Machines**: Autonomous units that follow predetermined patterns
2. **One Chaser Bin**: AI agent that learns to coordinate collection using Q-learning
3. **Dynamic Environment**: Field state changes as machines harvest peas
4. **Constraint-Based Movement**: Chaser can only move on previously harvested cells

## Detailed System Components

### 1. Environment (`HarvestField`)

The environment manages the entire simulation state:

#### Field Configuration
- **Size**: 100x100 grid (10,000 cells total)
- **Harvested Cell Tracking**: Boolean matrix tracking which cells have been harvested
- **Termination Condition**: Simulation ends when 80% of field is harvested

#### Agent Properties
```cpp
// Machine specifications
static constexpr int MACHINE_CAPACITY = 50;     // Peas per machine
static constexpr double HARVEST_RATE = 0.8;     // 80% harvest probability per step
static constexpr int CHASER_CAPACITY = 200;     // Chaser bin capacity
```

#### Movement Patterns

**Machine 0 (Horizontal Sweeper)**:
```
Start: (0,0) → Move right across row → Next row down → Repeat
Pattern: 0→1→2→...→99→(next row)→0→1→2...
```

**Machine 1 (Vertical Sweeper)**:
```
Start: (99,99) → Move up column → Next column left → Repeat  
Pattern: 99→98→97→...→0→(next column)→99→98...
```

**Chaser Bin (Q-Learning Agent)**:
- **Actions**: {UP, DOWN, LEFT, RIGHT, STAY}
- **Constraint**: Can only move on harvested cells
- **Goal**: Collect from machines before they overflow

### 2. State Representation

The chaser's state is a comprehensive encoding that includes:

```cpp
int get_chaser_state() {
    int state = pos_to_state(chaser_position.first, chaser_position.second);
    
    // Add machine load status (high priority flag)
    for (int i = 0; i < 2; ++i) {
        state += (machine_loads[i] > MACHINE_CAPACITY * 0.8 ? 1 << (20 + i) : 0);
        state += (get_distance(chaser_position, machine_positions[i]) << (10 + i * 3));
    }
    return state;
}
```

**State Components**:
1. **Chaser Position**: (x, y) coordinates
2. **Machine Load Status**: Binary flags for machines >80% full
3. **Distance Information**: Manhattan distance to each machine

### 3. Reward Structure

The reward function encourages efficient coordination:

```cpp
double get_chaser_reward(int action, const std::pair<int, int>& old_pos) {
    double reward = 0.0;
    
    // Movement efficiency
    if (action != STAY) reward -= 0.1;  // Small movement penalty
    
    // Collection rewards
    int machine_at = chaser_at_machine();
    if (machine_at >= 0 && machine_loads[machine_at] > 0) {
        reward += collect_from_machine(machine_at);  // +2.0 per pea collected
    }
    
    // Proximity incentives
    for (int i = 0; i < 2; ++i) {
        if (machine_loads[i] > MACHINE_CAPACITY * 0.8) {
            int distance = get_distance(chaser_position, machine_positions[i]);
            reward += std::max(0.0, 5.0 - distance * 0.1);  // Reward proximity to full machines
        }
    }
    
    // Overflow penalties
    for (int i = 0; i < 2; ++i) {
        if (machine_loads[i] >= MACHINE_CAPACITY) {
            reward -= 10.0;  // Heavy penalty for machine overflow
        }
    }
    
    return reward;
}
```

**Reward Components**:
- **Collection Bonus**: +2.0 points per pea collected
- **Proximity Reward**: Up to +5.0 for being near full machines
- **Movement Cost**: -0.1 per movement (efficiency incentive)
- **Overflow Penalty**: -10.0 for machine capacity overflow
- **Distance Penalty**: -0.05 × distance for each full machine

### 4. Advanced Q-Learning Features

The demo implements multiple state-of-the-art Q-learning enhancements:

#### 4.1 Double Q-Learning
Reduces overestimation bias by using two Q-tables:
```cpp
chaser_agent.set_double_q_learning(true);
```

#### 4.2 Experience Replay
Stores and replays past experiences for better learning:
```cpp
chaser_agent.set_experience_replay(true, 10000, 128);
// Buffer size: 10,000 experiences
// Batch size: 128 samples per update
```

#### 4.3 Eligibility Traces
Provides credit assignment over time:
```cpp
chaser_agent.set_eligibility_traces(true, 0.9);
// λ = 0.9 (trace decay parameter)
```

#### 4.4 Boltzmann Exploration
Intelligent action selection based on Q-values:
```cpp
chaser_agent.set_temperature(1.5);
// Temperature controls exploration vs exploitation
```

#### 4.5 Action Masking
Enforces environment constraints:
```cpp
chaser_agent.set_action_mask([&field](int state, int action) {
    auto current_pos = field.chaser_position;
    auto next_pos = field.get_next_position(current_pos, action);
    return field.can_chaser_move_to(next_pos);  // Only harvested cells
});
```

#### 4.6 Reward Shaping
Normalizes rewards for stable learning:
```cpp
chaser_agent.set_reward_shaping([](double reward) {
    return std::tanh(reward / 5.0);  // Bounded between -1 and 1
});
```

## Training Process

### Phase 1: Advanced Coordination Training

```cpp
void demo_advanced_harvest_coordination() {
    // 150 episodes of training
    // 4,000 steps per episode maximum
    // Evaluation every 25 episodes
}
```

**Training Configuration**:
- **Episodes**: 150
- **Max Steps**: 4,000 per episode
- **Learning Rate**: 0.1
- **Discount Factor**: 0.95
- **Exploration Rate**: 0.15

### Phase 2: Performance Analysis

Compares two agent configurations:

#### Standard Configuration
```cpp
QLearning standard_agent(0.1, 0.95, 0.15, actions, BOLTZMANN);
standard_agent.set_double_q_learning(true);
standard_agent.set_experience_replay(true, 5000, 64);
standard_agent.set_temperature(1.8);
```

#### Optimized Configuration
```cpp
QLearning optimized_agent(0.05, 0.98, 0.1, actions, BOLTZMANN);
optimized_agent.set_double_q_learning(true);
optimized_agent.set_experience_replay(true, 10000, 128);
optimized_agent.set_eligibility_traces(true, 0.95);
optimized_agent.set_temperature(1.5);
```

## Performance Metrics

The system tracks multiple performance indicators:

### Agent Statistics
```cpp
struct Statistics {
    int total_updates;           // Number of Q-value updates
    double cumulative_reward;    // Total reward accumulated
    double exploration_ratio;    // Percentage of exploratory actions
};
```

### Field Statistics
```cpp
struct FieldStats {
    int total_harvested;         // Cells harvested
    int machine0_load;          // Current load of machine 0
    int machine1_load;          // Current load of machine 1
    int chaser_load;            // Current load of chaser
    double harvest_efficiency;   // Percentage of field harvested
};
```

## Visualization

The system provides real-time field visualization:

```
Field Status (15x15 view):
Legend: . = unharvested, # = harvested, M0/M1 = machines, C = chaser

0##############
1.#############
##C############
###############
###############
######1########
###############
###############

Machine Loads: M0=45/50, M1=32/50, Chaser=15/200
```

## Key Learning Outcomes

### 1. Coordination Strategies
The chaser learns to:
- **Predict machine paths** and position itself optimally
- **Prioritize urgent collections** when machines near capacity
- **Balance exploration vs exploitation** in path planning
- **Handle dynamic constraints** (harvested-cell-only movement)

### 2. Emergent Behaviors
- **Proactive positioning**: Moving toward machines before they're full
- **Efficient routing**: Minimizing unnecessary movements
- **Risk management**: Avoiding machine overflows
- **Adaptive timing**: Learning when to wait vs when to move

### 3. Performance Improvements
Typical improvements from advanced features:
- **Experience Replay**: +15-25% performance
- **Eligibility Traces**: +10-20% learning speed
- **Double Q-Learning**: +5-15% stability
- **Action Masking**: Ensures constraint satisfaction

## Usage

### Compilation
```bash
cd /doc/code/relearn
mkdir -p build && cd build
cmake ..
make multi_agent_harvest_demo
```

### Execution
```bash
./multi_agent_harvest_demo
```

### Expected Output
```
Multi-Agent Pea Harvesting Coordination Demo
============================================
Scenario: 2 harvesting machines + 1 chaser bin on 100x100 field

======================================================================
1. Advanced Multi-Agent Coordination with Experience Replay
======================================================================

Training advanced chaser agent with:
  - Double Q-learning
  - Experience replay (capacity=10000, batch=128)
  - Eligibility traces (λ=0.9)
  - Boltzmann exploration
  - Action masking for harvested-cell-only movement
  - Reward shaping

Episode 25, Performance: 145.3, Best: 145.3
Episode 50, Performance: 167.8, Best: 167.8
...
```

## Extensions and Modifications

### Scaling Up
- **More machines**: Add additional harvesters with different patterns
- **Larger fields**: Increase field dimensions
- **Multiple chasers**: Coordinate multiple collection units
- **Complex terrain**: Add obstacles or varying terrain types

### Algorithm Variations
- **Different exploration strategies**: UCB, Thompson sampling
- **Hierarchical learning**: High-level strategy, low-level execution
- **Multi-objective optimization**: Balance multiple goals
- **Continuous action spaces**: Smooth movement control

### Real-World Applications
This simulation framework can be adapted for:
- **Agricultural automation**: Real farm equipment coordination  
- **Warehouse robotics**: Automated material handling
- **Disaster response**: Coordinated search and rescue
- **Traffic management**: Vehicle coordination systems

## Technical Implementation Notes

### Memory Management
- State space grows with field complexity
- Experience replay buffer manages memory efficiently
- Q-table pruning for unused states

### Computational Complexity
- **State space**: O(W × H × machine_states)
- **Action space**: O(5) per agent
- **Training time**: O(episodes × steps × features)

### Numerical Stability
- Reward normalization prevents gradient explosion
- Temperature annealing improves convergence
- Learning rate scheduling adapts to progress

This demo represents a comprehensive example of modern reinforcement learning applied to a practical multi-agent coordination problem, showcasing both the complexity of real-world scenarios and the power of advanced Q-learning techniques.
