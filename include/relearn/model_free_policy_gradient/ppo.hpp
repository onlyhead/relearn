#pragma once

#include <algorithm>
#include <memory>
#include <vector>

namespace relearn {
    namespace model_free_policy_gradient {

        /**
         * @brief Proximal Policy Optimization (PPO) implementation
         *
         * Simplified approximation to TRPO using clipped surrogate objective.
         * Widely used for robotics due to stability and computational efficiency.
         */
        template <typename StateType, typename ActionType> class PPO {
          private:
            // Policy network, value network, and clipping parameters
            double clip_ratio_ = 0.2;
            int update_epochs_ = 4;

          public:
            inline PPO() = default;

            // Method signatures to be implemented
            inline void collect_rollouts();
            inline void update_policy();
            inline void update_value_function();
            inline ActionType sample_action(const StateType &state);
            inline double compute_clipped_surrogate_loss();
            inline double compute_advantage(const std::vector<double> &rewards, const std::vector<double> &values);
        };

    } // namespace model_free_policy_gradient
} // namespace relearn
