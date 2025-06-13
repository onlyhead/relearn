#pragma once

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace relearn {
    namespace model_free_actor_critic {

        /**
         * @brief Soft Actor-Critic (SAC) implementation
         *
         * Off-policy maximum entropy RL algorithm.
         * Excellent for continuous control with automatic temperature tuning.
         */
        template <typename StateType, typename ActionType> class SAC {
          private:
            // Actor and twin critic networks
            double temperature_ = 0.2;
            bool auto_tune_temperature_ = true;
            double target_entropy_;

          public:
            inline SAC() = default;

            // Method signatures to be implemented
            inline void update(const StateType &state, const ActionType &action, double reward,
                               const StateType &next_state, bool done);
            inline ActionType sample_action(const StateType &state, bool deterministic = false);
            inline void train_step();
            inline double compute_entropy_loss();
            inline void update_temperature();
            inline double reparameterize_action(const StateType &state);
        };

    } // namespace model_free_actor_critic
} // namespace relearn
