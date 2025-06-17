#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace relearn {
    namespace model_free_value_based {

        /**
         * @brief SARSA (State-Action-Reward-State-Action) Algorithm
         *
         * On-policy temporal difference learning algorithm that learns the value
         * of the policy being followed, rather than the optimal policy.
         * SARSA is more conservative and risk-aware compared to Q-Learning.
         *
         * Features:
         * - On-policy learning with policy consistency
         * - Eligibility traces (SARSA(λ))
         * - Multiple exploration strategies
         * - Learning rate scheduling
         * - Performance monitoring
         * - Thread-safe operations
         * - Experience replay (adapted for on-policy)
         * - Action masking and reward shaping
         */
        template <typename StateType, typename ActionType> class SARSA {
          public:
            // Exploration strategies
            enum class ExplorationStrategy { EPSILON_GREEDY, BOLTZMANN, UCB1, EPSILON_DECAY };

            // Learning rate schedules
            enum class LearningRateSchedule { CONSTANT, LINEAR_DECAY, EXPONENTIAL_DECAY, ADAPTIVE };

            // Experience for on-policy replay
            struct Experience {
                StateType state;
                ActionType action;
                double reward;
                StateType next_state;
                ActionType next_action;
                bool terminal;
                double importance_weight = 1.0;

                Experience(const StateType &s, const ActionType &a, double r, 
                          const StateType &ns, const ActionType &na, bool term)
                    : state(s), action(a), reward(r), next_state(ns), next_action(na), terminal(term) {}
            };

            // Statistics tracking
            struct Statistics {
                size_t total_updates = 0;
                size_t total_actions = 0;
                double cumulative_reward = 0.0;
                double average_q_value = 0.0;
                double exploration_ratio = 0.0;
                std::chrono::milliseconds total_training_time{0};

                void reset() {
                    total_updates = 0;
                    total_actions = 0;
                    cumulative_reward = 0.0;
                    average_q_value = 0.0;
                    exploration_ratio = 0.0;
                    total_training_time = std::chrono::milliseconds{0};
                }
            };

          private:
            // Core Q-table
            std::unordered_map<StateType, std::unordered_map<ActionType, double>> q_table_;

            // Eligibility traces
            std::unordered_map<StateType, std::unordered_map<ActionType, double>> eligibility_traces_;

            // Visit counts for UCB exploration
            std::unordered_map<StateType, std::unordered_map<ActionType, size_t>> visit_counts_;
            std::unordered_map<StateType, size_t> state_visit_counts_;

            // Experience replay buffer (adapted for on-policy)
            std::vector<Experience> replay_buffer_;
            size_t replay_buffer_capacity_ = 10000;
            size_t replay_batch_size_ = 32;

            // Learning parameters
            double alpha_ = 0.1;       // Learning rate
            double gamma_ = 0.95;      // Discount factor
            double epsilon_ = 0.1;     // Exploration rate
            double lambda_ = 0.0;      // Eligibility trace decay
            double temperature_ = 1.0; // Boltzmann temperature
            double ucb_c_ = 2.0;       // UCB exploration parameter

            // Advanced parameters
            bool use_eligibility_traces_ = false;
            bool use_experience_replay_ = false;
            double min_epsilon_ = 0.01;
            double epsilon_decay_ = 0.995;
            double alpha_decay_ = 1.0;
            double min_alpha_ = 0.001;

            // Configuration
            ExplorationStrategy exploration_strategy_ = ExplorationStrategy::EPSILON_GREEDY;
            LearningRateSchedule lr_schedule_ = LearningRateSchedule::CONSTANT;
            std::vector<ActionType> available_actions_;
            std::function<bool(const StateType &, const ActionType &)> action_mask_;
            std::function<double(double)> reward_shaping_;

            // Random number generation
            mutable std::mt19937 rng_;
            mutable std::uniform_real_distribution<double> uniform_dist_;
            mutable std::uniform_int_distribution<size_t> action_dist_;

            // Thread safety
            mutable std::mutex table_mutex_;

            // Statistics
            Statistics stats_;

          public:
            /**
             * @brief Constructor with comprehensive configuration
             */
            inline SARSA(double alpha = 0.1, double gamma = 0.95, double epsilon = 0.1,
                        const std::vector<ActionType> &actions = {},
                        ExplorationStrategy exploration = ExplorationStrategy::EPSILON_GREEDY,
                        LearningRateSchedule lr_schedule = LearningRateSchedule::CONSTANT)
                : alpha_(alpha), gamma_(gamma), epsilon_(epsilon), exploration_strategy_(exploration),
                  lr_schedule_(lr_schedule), available_actions_(actions), rng_(std::random_device{}()),
                  uniform_dist_(0.0, 1.0) {

                if (!actions.empty()) {
                    action_dist_ = std::uniform_int_distribution<size_t>(0, actions.size() - 1);
                }
            }

            /**
             * @brief SARSA update with current and next action
             * Key difference from Q-Learning: uses actual next action, not max
             */
            inline void update(const StateType &state, const ActionType &action, double reward,
                              const StateType &next_state, const ActionType &next_action, 
                              bool terminal = false) {
                std::lock_guard<std::mutex> lock(table_mutex_);

                auto start_time = std::chrono::high_resolution_clock::now();

                // Apply reward shaping if configured
                if (reward_shaping_) {
                    reward = reward_shaping_(reward);
                }

                // SARSA update: Q(s,a) += α[r + γQ(s',a') - Q(s,a)]
                double current_q = get_q_value_internal(state, action);
                double next_q = terminal ? 0.0 : get_q_value_internal(next_state, next_action);
                double td_error = reward + gamma_ * next_q - current_q;
                
                q_table_[state][action] = current_q + alpha_ * td_error;

                // Update eligibility traces if enabled
                if (use_eligibility_traces_) {
                    update_eligibility_traces(state, action, td_error);
                }

                // Store experience for replay if enabled
                if (use_experience_replay_) {
                    store_experience(state, action, reward, next_state, next_action, terminal);
                }

                // Update learning rate schedule
                update_learning_rate();

                // Update statistics
                stats_.total_updates++;
                stats_.cumulative_reward += reward;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                stats_.total_training_time += duration;
            }

            /**
             * @brief Select action using the same policy for consistency
             */
            inline ActionType select_action(const StateType &state) {
                if (available_actions_.empty()) {
                    throw std::runtime_error("No available actions set. Use set_actions() first.");
                }

                std::lock_guard<std::mutex> lock(table_mutex_);

                // Get valid actions (apply action masking if configured)
                std::vector<ActionType> valid_actions;
                for (const auto &action : available_actions_) {
                    if (!action_mask_ || action_mask_(state, action)) {
                        valid_actions.push_back(action);
                    }
                }

                if (valid_actions.empty()) {
                    throw std::runtime_error("No valid actions available for current state.");
                }

                ActionType selected_action;
                bool explored = false;

                switch (exploration_strategy_) {
                case ExplorationStrategy::EPSILON_GREEDY:
                    selected_action = epsilon_greedy_selection(state, valid_actions, explored);
                    break;
                case ExplorationStrategy::BOLTZMANN:
                    selected_action = boltzmann_selection(state, valid_actions, explored);
                    break;
                case ExplorationStrategy::UCB1:
                    selected_action = ucb_selection(state, valid_actions, explored);
                    break;
                case ExplorationStrategy::EPSILON_DECAY:
                    selected_action = epsilon_decay_selection(state, valid_actions, explored);
                    break;
                }

                // Update visit counts
                visit_counts_[state][selected_action]++;
                state_visit_counts_[state]++;

                // Update statistics
                stats_.total_actions++;
                if (explored) {
                    stats_.exploration_ratio =
                        (stats_.exploration_ratio * (stats_.total_actions - 1) + 1.0) / stats_.total_actions;
                } else {
                    stats_.exploration_ratio =
                        (stats_.exploration_ratio * (stats_.total_actions - 1)) / stats_.total_actions;
                }

                return selected_action;
            }

            /**
             * @brief Experience replay for on-policy learning
             * Note: Only replays experiences that are consistent with current policy
             */
            inline void replay_experience() {
                if (!use_experience_replay_ || replay_buffer_.size() < replay_batch_size_) {
                    return;
                }

                std::lock_guard<std::mutex> lock(table_mutex_);

                // Sample recent experiences (more on-policy consistent)
                std::vector<Experience> batch;
                size_t start_idx = replay_buffer_.size() >= replay_batch_size_ ? 
                                  replay_buffer_.size() - replay_batch_size_ : 0;
                
                for (size_t i = start_idx; i < replay_buffer_.size(); ++i) {
                    batch.push_back(replay_buffer_[i]);
                }

                // Train on batch
                for (const auto &exp : batch) {
                    double current_q = get_q_value_internal(exp.state, exp.action);
                    double next_q = exp.terminal ? 0.0 : get_q_value_internal(exp.next_state, exp.next_action);
                    double td_error = exp.reward + gamma_ * next_q - current_q;
                    
                    q_table_[exp.state][exp.action] = current_q + alpha_ * td_error * exp.importance_weight;
                }
            }

            /**
             * @brief Get Q-value for state-action pair
             */
            inline double get_q_value(const StateType &state, const ActionType &action) const {
                std::lock_guard<std::mutex> lock(table_mutex_);
                return get_q_value_internal(state, action);
            }

            /**
             * @brief Get maximum Q-value for a state
             */
            inline double get_max_q_value(const StateType &state) const {
                std::lock_guard<std::mutex> lock(table_mutex_);
                return get_max_q_value_internal(state);
            }

            /**
             * @brief Get best action for a state (greedy policy)
             */
            inline ActionType get_best_action(const StateType &state) const {
                if (available_actions_.empty()) {
                    throw std::runtime_error("No available actions set.");
                }

                std::lock_guard<std::mutex> lock(table_mutex_);

                ActionType best_action = available_actions_[0];
                double best_q = get_q_value_internal(state, best_action);

                for (const auto &action : available_actions_) {
                    if (!action_mask_ || action_mask_(state, action)) {
                        double q_val = get_q_value_internal(state, action);
                        if (q_val > best_q) {
                            best_q = q_val;
                            best_action = action;
                        }
                    }
                }

                return best_action;
            }

            /**
             * @brief Configuration methods
             */
            inline void set_actions(const std::vector<ActionType> &actions) {
                available_actions_ = actions;
                if (!actions.empty()) {
                    action_dist_ = std::uniform_int_distribution<size_t>(0, actions.size() - 1);
                }
            }

            inline void set_action_mask(std::function<bool(const StateType &, const ActionType &)> mask) {
                action_mask_ = mask;
            }

            inline void set_reward_shaping(std::function<double(double)> shaping) {
                reward_shaping_ = shaping;
            }

            inline void enable_eligibility_traces(double lambda) {
                use_eligibility_traces_ = true;
                lambda_ = lambda;
            }

            inline void enable_experience_replay(size_t capacity = 10000, size_t batch_size = 32) {
                use_experience_replay_ = true;
                replay_buffer_capacity_ = capacity;
                replay_batch_size_ = batch_size;
                replay_buffer_.reserve(capacity);
            }

            inline void set_exploration_strategy(ExplorationStrategy strategy) {
                exploration_strategy_ = strategy;
            }

            inline void set_learning_rate_schedule(LearningRateSchedule schedule) {
                lr_schedule_ = schedule;
            }

            /**
             * @brief Parameter access methods
             */
            inline double get_alpha() const { return alpha_; }
            inline double get_gamma() const { return gamma_; }
            inline double get_epsilon() const { return epsilon_; }
            inline double get_lambda() const { return lambda_; }
            inline const Statistics& get_statistics() const { return stats_; }

            inline void set_alpha(double alpha) { alpha_ = std::max(alpha, min_alpha_); }
            inline void set_gamma(double gamma) { gamma_ = std::clamp(gamma, 0.0, 1.0); }
            inline void set_epsilon(double epsilon) { epsilon_ = std::clamp(epsilon, min_epsilon_, 1.0); }
            inline void set_lambda(double lambda) { lambda_ = std::clamp(lambda, 0.0, 1.0); }

            /**
             * @brief Reset and persistence methods
             */
            inline void reset() {
                std::lock_guard<std::mutex> lock(table_mutex_);
                q_table_.clear();
                eligibility_traces_.clear();
                visit_counts_.clear();
                state_visit_counts_.clear();
                replay_buffer_.clear();
                stats_.reset();
            }

            inline void save_policy(const std::string &filename) const {
                std::lock_guard<std::mutex> lock(table_mutex_);
                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                for (const auto &state_entry : q_table_) {
                    for (const auto &action_entry : state_entry.second) {
                        file << state_entry.first << " " << action_entry.first 
                             << " " << action_entry.second << "\n";
                    }
                }
            }

            inline void load_policy(const std::string &filename) {
                std::lock_guard<std::mutex> lock(table_mutex_);
                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for reading: " + filename);
                }

                q_table_.clear();
                StateType state;
                ActionType action;
                double q_value;
                
                while (file >> state >> action >> q_value) {
                    q_table_[state][action] = q_value;
                }
            }

          private:
            /**
             * @brief Internal helper methods
             */
            inline double get_q_value_internal(const StateType &state, const ActionType &action) const {
                auto state_it = q_table_.find(state);
                if (state_it != q_table_.end()) {
                    auto action_it = state_it->second.find(action);
                    if (action_it != state_it->second.end()) {
                        return action_it->second;
                    }
                }
                return 0.0; // Default Q-value for unseen state-action pairs
            }

            inline double get_max_q_value_internal(const StateType &state) const {
                if (available_actions_.empty()) {
                    return 0.0;
                }

                double max_q = std::numeric_limits<double>::lowest();
                bool found_valid = false;

                for (const auto &action : available_actions_) {
                    if (!action_mask_ || action_mask_(state, action)) {
                        double q_val = get_q_value_internal(state, action);
                        if (!found_valid || q_val > max_q) {
                            max_q = q_val;
                            found_valid = true;
                        }
                    }
                }

                return found_valid ? max_q : 0.0;
            }

            inline ActionType epsilon_greedy_selection(const StateType &state, 
                                                      const std::vector<ActionType> &valid_actions, 
                                                      bool &explored) const {
                if (uniform_dist_(rng_) < epsilon_) {
                    explored = true;
                    std::uniform_int_distribution<size_t> dist(0, valid_actions.size() - 1);
                    return valid_actions[dist(rng_)];
                } else {
                    explored = false;
                    ActionType best_action = valid_actions[0];
                    double best_q = get_q_value_internal(state, best_action);

                    for (const auto &action : valid_actions) {
                        double q_val = get_q_value_internal(state, action);
                        if (q_val > best_q) {
                            best_q = q_val;
                            best_action = action;
                        }
                    }
                    return best_action;
                }
            }

            inline ActionType epsilon_decay_selection(const StateType &state, 
                                                     const std::vector<ActionType> &valid_actions, 
                                                     bool &explored) const {
                // Use current epsilon for this selection
                if (uniform_dist_(rng_) < epsilon_) {
                    explored = true;
                    std::uniform_int_distribution<size_t> dist(0, valid_actions.size() - 1);
                    return valid_actions[dist(rng_)];
                } else {
                    explored = false;
                    return get_greedy_action(state, valid_actions);
                }
            }

            inline ActionType boltzmann_selection(const StateType &state, 
                                                 const std::vector<ActionType> &valid_actions, 
                                                 bool &explored) const {
                std::vector<double> probabilities;
                double sum_exp = 0.0;

                // Calculate unnormalized probabilities
                for (const auto &action : valid_actions) {
                    double q_val = get_q_value_internal(state, action);
                    double exp_val = std::exp(q_val / temperature_);
                    probabilities.push_back(exp_val);
                    sum_exp += exp_val;
                }

                // Normalize probabilities
                for (auto &prob : probabilities) {
                    prob /= sum_exp;
                }

                // Sample according to probabilities
                double rand_val = uniform_dist_(rng_);
                double cumulative = 0.0;
                
                for (size_t i = 0; i < valid_actions.size(); ++i) {
                    cumulative += probabilities[i];
                    if (rand_val <= cumulative) {
                        explored = (probabilities[i] < 0.9); // Arbitrary threshold for "exploration"
                        return valid_actions[i];
                    }
                }

                // Fallback (shouldn't happen with proper implementation)
                explored = false;
                return valid_actions.back();
            }

            inline ActionType ucb_selection(const StateType &state, 
                                           const std::vector<ActionType> &valid_actions, 
                                           bool &explored) const {
                ActionType best_action = valid_actions[0];
                double best_ucb = std::numeric_limits<double>::lowest();
                size_t total_visits = state_visit_counts_.find(state) != state_visit_counts_.end() ? 
                                     state_visit_counts_.at(state) : 0;

                for (const auto &action : valid_actions) {
                    double q_val = get_q_value_internal(state, action);
                    size_t action_visits = 0;
                    
                    auto state_it = visit_counts_.find(state);
                    if (state_it != visit_counts_.end()) {
                        auto action_it = state_it->second.find(action);
                        if (action_it != state_it->second.end()) {
                            action_visits = action_it->second;
                        }
                    }

                    double ucb_bonus = action_visits > 0 ? 
                                      ucb_c_ * std::sqrt(std::log(total_visits) / action_visits) : 
                                      std::numeric_limits<double>::max();
                    
                    double ucb_value = q_val + ucb_bonus;
                    
                    if (ucb_value > best_ucb) {
                        best_ucb = ucb_value;
                        best_action = action;
                    }
                }

                explored = (best_ucb == std::numeric_limits<double>::max()); // High bonus means exploration
                return best_action;
            }

            inline ActionType get_greedy_action(const StateType &state, 
                                               const std::vector<ActionType> &valid_actions) const {
                ActionType best_action = valid_actions[0];
                double best_q = get_q_value_internal(state, best_action);

                for (const auto &action : valid_actions) {
                    double q_val = get_q_value_internal(state, action);
                    if (q_val > best_q) {
                        best_q = q_val;
                        best_action = action;
                    }
                }
                return best_action;
            }

            inline void update_eligibility_traces(const StateType &state, const ActionType &action, double td_error) {
                // Decay all traces
                for (auto &state_entry : eligibility_traces_) {
                    for (auto &action_entry : state_entry.second) {
                        action_entry.second *= gamma_ * lambda_;
                    }
                }

                // Set current state-action trace to 1
                eligibility_traces_[state][action] = 1.0;

                // Update all Q-values proportional to their traces
                for (const auto &state_entry : eligibility_traces_) {
                    for (const auto &action_entry : state_entry.second) {
                        if (action_entry.second > 0.001) { // Only update significant traces
                            q_table_[state_entry.first][action_entry.first] += 
                                alpha_ * td_error * action_entry.second;
                        }
                    }
                }
            }

            inline void store_experience(const StateType &state, const ActionType &action, double reward,
                                        const StateType &next_state, const ActionType &next_action, bool terminal) {
                if (replay_buffer_.size() >= replay_buffer_capacity_) {
                    // Remove oldest experience (circular buffer)
                    replay_buffer_.erase(replay_buffer_.begin());
                }
                
                replay_buffer_.emplace_back(state, action, reward, next_state, next_action, terminal);
            }

            inline void update_learning_rate() {
                switch (lr_schedule_) {
                case LearningRateSchedule::CONSTANT:
                    // No change
                    break;
                case LearningRateSchedule::LINEAR_DECAY:
                    alpha_ = std::max(alpha_ * alpha_decay_, min_alpha_);
                    break;
                case LearningRateSchedule::EXPONENTIAL_DECAY:
                    alpha_ = std::max(alpha_ * std::exp(-alpha_decay_ * stats_.total_updates / 1000.0), min_alpha_);
                    break;
                case LearningRateSchedule::ADAPTIVE:
                    // Simple adaptive rule based on performance
                    if (stats_.total_updates > 100 && stats_.total_updates % 100 == 0) {
                        double recent_avg_reward = stats_.cumulative_reward / stats_.total_updates;
                        if (recent_avg_reward < -10.0) {
                            alpha_ = std::min(alpha_ * 1.05, 0.5); // Increase if performing poorly
                        } else {
                            alpha_ = std::max(alpha_ * 0.99, min_alpha_); // Decrease if doing well
                        }
                    }
                    break;
                }
                
                // Handle epsilon decay if using epsilon decay strategy
                if (exploration_strategy_ == ExplorationStrategy::EPSILON_DECAY) {
                    epsilon_ = std::max(epsilon_ * epsilon_decay_, min_epsilon_);
                }
            }
        };

    } // namespace model_free_value_based
} // namespace relearn
