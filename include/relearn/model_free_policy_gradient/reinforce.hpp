#pragma once

#include <memory>
#include <vector>

namespace relearn {
    namespace model_free_policy_gradient {

        /**
         * @brief REINFORCE algorithm implementation
         *
         * Monte-Carlo policy gradient method.
         * Updates policy parameters using full episode returns.
         */
        template <typename StateType, typename ActionType> class REINFORCE {
          private:
            // Policy network and trajectory storage

          public:
            inline REINFORCE() = default;

            // Method signatures to be implemented
            inline void store_transition(const StateType &state, const ActionType &action, double reward);
            inline void update_policy();
            inline ActionType sample_action(const StateType &state);
            inline void reset_episode();
        };

    } // namespace model_free_policy_gradient
} // namespace relearn
