#pragma once

#include <memory>
#include <vector>

namespace relearn {
    namespace model_based {

        /**
         * @brief Probabilistic Inference for Learning Control (PILCO) implementation
         *
         * Uses Gaussian processes for dynamics modeling.
         * Extremely sample efficient but computationally expensive.
         */
        template <typename StateType, typename ActionType> class PILCO {
          private:
            // Gaussian process for dynamics model
            // Policy parameterization

          public:
            inline PILCO() = default;

            // Method signatures to be implemented
            inline void fit_dynamics_model(const std::vector<StateType> &states, const std::vector<ActionType> &actions,
                                           const std::vector<StateType> &next_states);
            inline void optimize_policy();
            inline ActionType select_action(const StateType &state);
            inline StateType predict_next_state(const StateType &state, const ActionType &action);
            inline double compute_expected_return();
        };

    } // namespace model_based
} // namespace relearn
