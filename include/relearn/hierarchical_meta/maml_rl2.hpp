#pragma once

#include <memory>
#include <vector>

namespace relearn {
    namespace hierarchical_meta {

        /**
         * @brief Model-Agnostic Meta-Learning (MAML) implementation
         *
         * Learns initialization parameters for rapid adaptation to new tasks.
         * Uses gradient-based meta-learning with few-shot adaptation.
         */
        template <typename StateType, typename ActionType> class MAML {
          private:
            // Meta-parameters (initialization)
            // Task distribution
            double meta_learning_rate_ = 0.001;
            double adaptation_learning_rate_ = 0.01;
            int adaptation_steps_ = 5;

          public:
            inline MAML() = default;

            // Method signatures to be implemented
            inline void meta_train(const std::vector<std::vector<StateType>> &task_batches);
            inline void adapt_to_task(const std::vector<StateType> &task_data);
            inline ActionType select_action(const StateType &state);
            inline void compute_meta_gradients();
            inline void update_meta_parameters();
        };

        /**
         * @brief RL^2 (RL Squared) implementation
         *
         * Recurrent network that learns to adapt its behavior based on experience.
         * Treats adaptation as a partially observable MDP.
         */
        template <typename StateType, typename ActionType> class RLSquared {
          private:
            // Recurrent neural network
            // Hidden state for memory across episodes

          public:
            inline RLSquared() = default;

            // Method signatures to be implemented
            inline void train_on_task_distribution(const std::vector<std::vector<StateType>> &tasks);
            inline ActionType select_action(const StateType &state, const std::vector<double> &hidden_state);
            inline std::vector<double> update_hidden_state(const StateType &state, const ActionType &action,
                                                           double reward, const std::vector<double> &prev_hidden);
            inline void reset_hidden_state();
        };

    } // namespace hierarchical_meta
} // namespace relearn
