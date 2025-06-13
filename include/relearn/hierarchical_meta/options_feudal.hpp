#pragma once

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace relearn {
    namespace hierarchical_meta {

        /**
         * @brief Options framework implementation
         *
         * Hierarchical RL with temporal abstractions.
         * Options are policies with initiation and termination conditions.
         */
        template <typename StateType, typename ActionType> class Options {
          private:
            // Set of available options
            // Option-value function
            // Intra-option policy

          public:
            struct Option {
                std::function<bool(const StateType &)> initiation_set;
                std::function<bool(const StateType &)> termination_condition;
                std::function<ActionType(const StateType &)> policy;
                int option_id;
            };

            inline Options() = default;

            // Method signatures to be implemented
            inline void add_option(const Option &option);
            inline int select_option(const StateType &state);
            inline ActionType select_action(const StateType &state, int option_id);
            inline void update_option_values();
            inline bool should_terminate_option(const StateType &state, int option_id);
        };

        /**
         * @brief Feudal Networks implementation
         *
         * Hierarchical RL with manager-worker architecture.
         * Manager sets goals for workers in a latent goal space.
         */
        template <typename StateType, typename ActionType> class FeudalNetworks {
          private:
            // Manager network (goal setting)
            // Worker network (primitive actions)
            int goal_horizon_ = 10;

          public:
            inline FeudalNetworks() = default;

            // Method signatures to be implemented
            inline std::vector<double> set_goal(const StateType &state);
            inline ActionType select_action(const StateType &state, const std::vector<double> &goal);
            inline void update_manager();
            inline void update_worker();
            inline double compute_intrinsic_reward(const StateType &state, const std::vector<double> &goal);
        };

    } // namespace hierarchical_meta
} // namespace relearn
