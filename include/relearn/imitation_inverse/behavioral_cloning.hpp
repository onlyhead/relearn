#pragma once

#include <memory>
#include <vector>

namespace relearn {
    namespace imitation_inverse {

        /**
         * @brief Behavioral Cloning (BC) implementation
         *
         * Supervised learning approach to imitation learning.
         * Learns policy directly from expert state-action pairs.
         */
        template <typename StateType, typename ActionType> class BehavioralCloning {
          private:
            // Policy network for supervised learning

          public:
            inline BehavioralCloning() = default;

            // Method signatures to be implemented
            inline void train(const std::vector<StateType> &expert_states,
                              const std::vector<ActionType> &expert_actions);
            inline ActionType predict_action(const StateType &state);
            inline double compute_loss(const std::vector<StateType> &states, const std::vector<ActionType> &actions);
        };

    } // namespace imitation_inverse
} // namespace relearn
