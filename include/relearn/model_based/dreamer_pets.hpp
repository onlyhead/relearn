#pragma once

#include <memory>
#include <vector>

namespace relearn {
    namespace model_based {

        /**
         * @brief Dreamer algorithm implementation
         *
         * Model-based RL with latent dynamics learning.
         * Uses world models for planning in latent space.
         */
        template <typename StateType, typename ActionType> class Dreamer {
          private:
            // World model components: encoder, dynamics, decoder, reward predictor
            // Actor-critic for policy learning in latent space

          public:
            inline Dreamer() = default;

            // Method signatures to be implemented
            inline void train_world_model();
            inline void train_policy_in_latent_space();
            inline ActionType select_action(const StateType &state);
            inline void encode_state(const StateType &state);
            inline void imagine_rollouts();
        };

        /**
         * @brief Probabilistic Ensembles with Trajectory Sampling (PETS) implementation
         *
         * Uses ensemble of probabilistic dynamics models for planning.
         * Performs random shooting or cross-entropy method for control.
         */
        template <typename StateType, typename ActionType> class PETS {
          private:
            // Ensemble of probabilistic neural networks
            // Planning algorithm (random shooting, CEM)
            int ensemble_size_ = 5;
            int planning_horizon_ = 15;

          public:
            inline PETS() = default;

            // Method signatures to be implemented
            inline void train_ensemble(const std::vector<StateType> &states, const std::vector<ActionType> &actions,
                                       const std::vector<StateType> &next_states);
            inline ActionType plan_action(const StateType &state);
            inline std::vector<ActionType> random_shooting(const StateType &state);
            inline std::vector<ActionType> cross_entropy_method(const StateType &state);
        };

    } // namespace model_based
} // namespace relearn
