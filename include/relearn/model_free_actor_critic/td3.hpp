#pragma once

#include "ddpg.hpp"
#include <algorithm>

namespace relearn {
    namespace model_free_actor_critic {

        /**
         * @brief Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation
         *
         * Improves DDPG stability by using twin critics and delayed policy updates.
         * Reduces overestimation bias through clipped double Q-learning.
         */
        template <typename StateType, typename ActionType> class TD3 : public DDPG<StateType, ActionType> {
          private:
            // Twin critic networks
            int policy_delay_ = 2;
            int update_counter_ = 0;
            double target_noise_clip_ = 0.5;

          public:
            inline TD3() = default;

            // Override methods for TD3-specific logic
            inline void train_step() override { /* TD3-specific implementation placeholder */ }
            inline double compute_target_q(const StateType &next_state, const ActionType &next_action) {
                (void)next_state;
                (void)next_action;
                return 0.0;
            }
            inline ActionType add_target_noise(const ActionType &action) {
                (void)action;
                return ActionType{};
            }
        };

    } // namespace model_free_actor_critic
} // namespace relearn
