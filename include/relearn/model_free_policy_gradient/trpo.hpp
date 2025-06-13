#pragma once

#include <cmath>
#include <memory>
#include <vector>

namespace relearn {
    namespace model_free_policy_gradient {

        /**
         * @brief Trust Region Policy Optimization (TRPO) implementation
         *
         * Constrains policy updates to stay within a trust region using KL divergence.
         * More stable than vanilla policy gradient but computationally expensive.
         */
        template <typename StateType, typename ActionType> class TRPO {
          private:
            // Policy network, value network, and trust region parameters

          public:
            inline TRPO() = default;

            // Method signatures to be implemented
            inline void collect_trajectories();
            inline void update_policy();
            inline void update_value_function();
            inline ActionType sample_action(const StateType &state);
            inline double compute_kl_divergence();
            inline void line_search();
        };

    } // namespace model_free_policy_gradient
} // namespace relearn
