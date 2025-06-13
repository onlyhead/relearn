#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace relearn {
    namespace model_free_value_based {

        /**
         * @brief Tabular Q-Learning implementation
         *
         * Classic Q-Learning for discrete state and action spaces.
         * Uses tabular representation with epsilon-greedy exploration.
         *
         * Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
         */
        template <typename StateType, typename ActionType> class QLearning {
          private:
            // Q-table: maps state to action-value pairs
            std::unordered_map<StateType, std::unordered_map<ActionType, double>> q_table_;

            // Learning parameters
            double alpha_ = 0.1;   // Learning rate
            double gamma_ = 0.95;  // Discount factor
            double epsilon_ = 0.1; // Exploration rate

            // Random number generator for epsilon-greedy selection
            mutable std::mt19937 rng_;
            mutable std::uniform_real_distribution<double> uniform_dist_;

            // Available actions (must be set by user)
            std::vector<ActionType> available_actions_;

          public:
            /**
             * @brief Constructor with default parameters
             */
            inline QLearning() : rng_(std::random_device{}()), uniform_dist_(0.0, 1.0) {}

            /**
             * @brief Constructor with custom parameters
             */
            inline QLearning(double alpha, double gamma, double epsilon, const std::vector<ActionType> &actions)
                : alpha_(alpha), gamma_(gamma), epsilon_(epsilon), rng_(std::random_device{}()),
                  uniform_dist_(0.0, 1.0), available_actions_(actions) {}

            /**
             * @brief Update Q-value using Q-learning rule
             * Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
             */
            inline void update(const StateType &state, const ActionType &action, double reward,
                               const StateType &next_state) {
                double current_q = get_q_value(state, action);
                double max_next_q = get_max_q_value(next_state);
                double target = reward + gamma_ * max_next_q;
                double new_q = current_q + alpha_ * (target - current_q);

                q_table_[state][action] = new_q;
            }

            /**
             * @brief Select action using epsilon-greedy policy
             */
            inline ActionType select_action(const StateType &state) {
                if (available_actions_.empty()) {
                    throw std::runtime_error("No available actions set. Use set_actions() first.");
                }

                // Epsilon-greedy exploration
                if (uniform_dist_(rng_) < epsilon_) {
                    // Random action (exploration)
                    std::uniform_int_distribution<size_t> action_dist(0, available_actions_.size() - 1);
                    return available_actions_[action_dist(rng_)];
                } else {
                    // Greedy action (exploitation)
                    return get_best_action(state);
                }
            }

            /**
             * @brief Get Q-value for state-action pair
             */
            inline double get_q_value(const StateType &state, const ActionType &action) const {
                auto state_it = q_table_.find(state);
                if (state_it == q_table_.end()) {
                    return 0.0; // Initialize to 0 if not found
                }

                auto action_it = state_it->second.find(action);
                if (action_it == state_it->second.end()) {
                    return 0.0; // Initialize to 0 if not found
                }

                return action_it->second;
            }

            /**
             * @brief Get maximum Q-value for a state
             */
            inline double get_max_q_value(const StateType &state) const {
                auto state_it = q_table_.find(state);
                if (state_it == q_table_.end() || state_it->second.empty()) {
                    return 0.0; // No actions recorded for this state
                }

                double max_q = std::numeric_limits<double>::lowest();
                for (const auto &action_value : state_it->second) {
                    max_q = std::max(max_q, action_value.second);
                }
                return max_q;
            }

            /**
             * @brief Get best action for a state (greedy policy)
             */
            inline ActionType get_best_action(const StateType &state) const {
                auto state_it = q_table_.find(state);
                if (state_it == q_table_.end() || state_it->second.empty()) {
                    // If no actions recorded, return random action
                    if (available_actions_.empty()) {
                        throw std::runtime_error("No available actions and no Q-values recorded.");
                    }
                    std::uniform_int_distribution<size_t> action_dist(0, available_actions_.size() - 1);
                    return available_actions_[action_dist(rng_)];
                }

                // Find action with maximum Q-value
                auto best_action = std::max_element(state_it->second.begin(), state_it->second.end(),
                                                    [](const auto &a, const auto &b) { return a.second < b.second; });

                return best_action->first;
            }

            // Setters for parameters
            inline void set_learning_rate(double alpha) { alpha_ = alpha; }
            inline void set_discount_factor(double gamma) { gamma_ = gamma; }
            inline void set_epsilon(double epsilon) { epsilon_ = epsilon; }
            inline void set_actions(const std::vector<ActionType> &actions) { available_actions_ = actions; }

            // Getters for parameters
            inline double get_learning_rate() const { return alpha_; }
            inline double get_discount_factor() const { return gamma_; }
            inline double get_epsilon() const { return epsilon_; }

            // Get the size of the Q-table (for debugging/monitoring)
            inline size_t get_q_table_size() const {
                size_t total_entries = 0;
                for (const auto &state_actions : q_table_) {
                    total_entries += state_actions.second.size();
                }
                return total_entries;
            }
        };

    } // namespace model_free_value_based
} // namespace relearn
