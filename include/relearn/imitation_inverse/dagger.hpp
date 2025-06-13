#pragma once

#include "behavioral_cloning.hpp"
#include <functional>

namespace relearn {
    namespace imitation_inverse {

        /**
         * @brief Dataset Aggregation (DAgger) implementation
         *
         * Iterative imitation learning that queries expert on visited states.
         * Addresses distribution shift problem in behavioral cloning.
         */
        template <typename StateType, typename ActionType>
        class DAgger : public BehavioralCloning<StateType, ActionType> {
          private:
            // Expert policy query function
            std::function<ActionType(const StateType &)> expert_policy_;
            double beta_ = 1.0; // Expert mixing parameter

          public:
            inline DAgger(std::function<ActionType(const StateType &)> expert_policy) : expert_policy_(expert_policy) {}

            // Method signatures to be implemented
            inline void train_iteration();
            inline void collect_trajectories();
            inline void query_expert_on_visited_states(const std::vector<StateType> &states);
            inline ActionType select_action(const StateType &state);
            inline void update_beta();
        };

    } // namespace imitation_inverse
} // namespace relearn
