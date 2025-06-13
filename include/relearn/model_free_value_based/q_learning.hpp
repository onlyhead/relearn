#pragma once

#include <algorithm>
#include <random>
#include <unordered_map>

namespace relearn {
    namespace model_free_value_based {

        /**
         * @brief Tabular Q-Learning implementation
         *
         * Classic Q-Learning for discrete state and action spaces.
         * Uses tabular representation with epsilon-greedy exploration.
         */
        template <typename StateType, typename ActionType> class QLearning {
          private:
            // Implementation will go here

          public:
            // Constructor and main methods will be implemented here
            inline QLearning() = default;

            // Method signatures to be implemented
            inline void update(const StateType &state, const ActionType &action, double reward,
                               const StateType &next_state);
            inline ActionType select_action(const StateType &state);
            inline double get_q_value(const StateType &state, const ActionType &action) const;
        };

    } // namespace model_free_value_based
} // namespace relearn
