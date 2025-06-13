#pragma once

#include "dqn.hpp"

namespace relearn {
namespace model_free_value_based {

/**
 * @brief Double DQN implementation
 * 
 * Reduces overestimation bias by using two Q-networks.
 * One network selects actions, the other evaluates them.
 */
template<typename StateType, typename ActionType>
class DoubleDQN : public DQN<StateType, ActionType> {
private:
    // Additional network for action evaluation

public:
    inline DoubleDQN() = default;
    
    // Override methods to implement Double DQN logic
    inline void train_step() override { /* Double DQN-specific implementation placeholder */ }
};

/**
 * @brief Dueling DQN implementation
 * 
 * Separates state value and advantage estimation in the network architecture.
 * Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
 */
template<typename StateType, typename ActionType>
class DuelingDQN : public DQN<StateType, ActionType> {
private:
    // Dueling network architecture

public:
    inline DuelingDQN() = default;
    
    // Override methods for dueling architecture
    inline double compute_q_value(const StateType& state, const ActionType& action);
};

} // namespace model_free_value_based
} // namespace relearn
