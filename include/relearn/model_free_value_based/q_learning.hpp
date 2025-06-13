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
         * @brief Production-Ready Q-Learning Implementation
         *
         * Features:
         * - Multiple exploration strategies (epsilon-greedy, Boltzmann, UCB)
         * - Eligibility traces (Q(λ))
         * - Learning rate scheduling
         * - Experience replay
         * - Q-table persistence
         * - Performance monitoring
         * - Thread-safe operations
         * - Action masking
         * - Reward shaping
         * - Double Q-learning variant
         */
        template <typename StateType, typename ActionType> class QLearning {
          public:
            // Exploration strategies
            enum class ExplorationStrategy { EPSILON_GREEDY, BOLTZMANN, UCB1, EPSILON_DECAY };

            // Learning rate schedules
            enum class LearningRateSchedule { CONSTANT, LINEAR_DECAY, EXPONENTIAL_DECAY, ADAPTIVE };

            // Experience for replay
            struct Experience {
                StateType state;
                ActionType action;
                double reward;
                StateType next_state;
                bool terminal;
                double importance_weight = 1.0;

                Experience(const StateType &s, const ActionType &a, double r, const StateType &ns, bool term)
                    : state(s), action(a), reward(r), next_state(ns), terminal(term) {}
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
            // Core Q-table and double Q-table for DDQN
            std::unordered_map<StateType, std::unordered_map<ActionType, double>> q_table_a_;
            std::unordered_map<StateType, std::unordered_map<ActionType, double>> q_table_b_;

            // Eligibility traces
            std::unordered_map<StateType, std::unordered_map<ActionType, double>> eligibility_traces_;

            // Visit counts for UCB exploration
            std::unordered_map<StateType, std::unordered_map<ActionType, size_t>> visit_counts_;
            std::unordered_map<StateType, size_t> state_visit_counts_;

            // Experience replay buffer
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
            bool use_double_q_learning_ = false;
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
            inline QLearning(double alpha = 0.1, double gamma = 0.95, double epsilon = 0.1,
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
             * @brief Update Q-value with advanced features
             */
            inline void update(const StateType &state, const ActionType &action, double reward,
                               const StateType &next_state, bool terminal = false) {
                std::lock_guard<std::mutex> lock(table_mutex_);

                auto start_time = std::chrono::high_resolution_clock::now();

                // Apply reward shaping if configured
                if (reward_shaping_) {
                    reward = reward_shaping_(reward);
                }

                if (use_double_q_learning_) {
                    update_double_q(state, action, reward, next_state, terminal);
                } else {
                    update_single_q(state, action, reward, next_state, terminal);
                }

                // Update eligibility traces if enabled
                if (use_eligibility_traces_) {
                    update_eligibility_traces(state, action, reward, next_state, terminal);
                }

                // Store experience for replay if enabled
                if (use_experience_replay_) {
                    store_experience(state, action, reward, next_state, terminal);
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
             * @brief Select action with advanced exploration strategies
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
             * @brief Experience replay training batch
             */
            inline void replay_experience() {
                if (!use_experience_replay_ || replay_buffer_.size() < replay_batch_size_) {
                    return;
                }

                std::lock_guard<std::mutex> lock(table_mutex_);

                // Sample random batch from replay buffer
                std::vector<Experience> batch;
                for (size_t i = 0; i < replay_batch_size_; ++i) {
                    std::uniform_int_distribution<size_t> dist(0, replay_buffer_.size() - 1);
                    batch.push_back(replay_buffer_[dist(rng_)]);
                }

                // Train on batch
                for (const auto &exp : batch) {
                    if (use_double_q_learning_) {
                        update_double_q(exp.state, exp.action, exp.reward, exp.next_state, exp.terminal);
                    } else {
                        update_single_q(exp.state, exp.action, exp.reward, exp.next_state, exp.terminal);
                    }
                }
            }

            /**
             * @brief Get Q-value with double Q-learning support
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
                return get_max_q_value_from_table(q_table_a_, state);
            }

            /**
             * @brief Get best action for a state
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
                        double q = get_q_value_internal(state, action);
                        if (q > best_q) {
                            best_q = q;
                            best_action = action;
                        }
                    }
                }

                return best_action;
            }

            /**
             * @brief Save Q-table to file
             */
            inline void save_q_table(const std::string &filename) const {
                std::lock_guard<std::mutex> lock(table_mutex_);

                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Save Q-table A
                size_t table_a_size = q_table_a_.size();
                file.write(reinterpret_cast<const char *>(&table_a_size), sizeof(table_a_size));

                for (const auto &state_pair : q_table_a_) {
                    // Write state (simplified - assumes POD types)
                    file.write(reinterpret_cast<const char *>(&state_pair.first), sizeof(StateType));

                    size_t actions_size = state_pair.second.size();
                    file.write(reinterpret_cast<const char *>(&actions_size), sizeof(actions_size));

                    for (const auto &action_pair : state_pair.second) {
                        file.write(reinterpret_cast<const char *>(&action_pair.first), sizeof(ActionType));
                        file.write(reinterpret_cast<const char *>(&action_pair.second), sizeof(double));
                    }
                }

                // Save statistics
                file.write(reinterpret_cast<const char *>(&stats_), sizeof(Statistics));

                file.close();
            }

            /**
             * @brief Load Q-table from file
             */
            inline void load_q_table(const std::string &filename) {
                std::lock_guard<std::mutex> lock(table_mutex_);

                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for reading: " + filename);
                }

                q_table_a_.clear();

                // Load Q-table A
                size_t table_a_size;
                file.read(reinterpret_cast<char *>(&table_a_size), sizeof(table_a_size));

                for (size_t i = 0; i < table_a_size; ++i) {
                    StateType state;
                    file.read(reinterpret_cast<char *>(&state), sizeof(StateType));

                    size_t actions_size;
                    file.read(reinterpret_cast<char *>(&actions_size), sizeof(actions_size));

                    for (size_t j = 0; j < actions_size; ++j) {
                        ActionType action;
                        double q_value;
                        file.read(reinterpret_cast<char *>(&action), sizeof(ActionType));
                        file.read(reinterpret_cast<char *>(&q_value), sizeof(double));

                        q_table_a_[state][action] = q_value;
                    }
                }

                // Load statistics
                file.read(reinterpret_cast<char *>(&stats_), sizeof(Statistics));

                file.close();
            }

            // Configuration methods
            inline void set_double_q_learning(bool enable) { use_double_q_learning_ = enable; }
            inline void set_eligibility_traces(bool enable, double lambda = 0.9) {
                use_eligibility_traces_ = enable;
                lambda_ = lambda;
            }
            inline void set_experience_replay(bool enable, size_t capacity = 10000, size_t batch_size = 32) {
                use_experience_replay_ = enable;
                replay_buffer_capacity_ = capacity;
                replay_batch_size_ = batch_size;
                if (enable) {
                    replay_buffer_.reserve(capacity);
                }
            }
            inline void set_action_mask(std::function<bool(const StateType &, const ActionType &)> mask) {
                action_mask_ = mask;
            }
            inline void set_reward_shaping(std::function<double(double)> shaping) { reward_shaping_ = shaping; }

            // Parameter setters
            inline void set_learning_rate(double alpha) { alpha_ = alpha; }
            inline void set_discount_factor(double gamma) { gamma_ = gamma; }
            inline void set_epsilon(double epsilon) { epsilon_ = epsilon; }
            inline void set_temperature(double temperature) { temperature_ = temperature; }
            inline void set_ucb_c(double ucb_c) { ucb_c_ = ucb_c; }
            inline void set_actions(const std::vector<ActionType> &actions) {
                available_actions_ = actions;
                if (!actions.empty()) {
                    action_dist_ = std::uniform_int_distribution<size_t>(0, actions.size() - 1);
                }
            }

            // Getters
            inline double get_learning_rate() const { return alpha_; }
            inline double get_discount_factor() const { return gamma_; }
            inline double get_epsilon() const { return epsilon_; }
            inline const Statistics &get_statistics() const { return stats_; }

            inline size_t get_q_table_size() const {
                std::lock_guard<std::mutex> lock(table_mutex_);
                size_t total = 0;
                for (const auto &state_pair : q_table_a_) {
                    total += state_pair.second.size();
                }
                return total;
            }

          private:
            // Implementation details

            /**
             * @brief Internal get Q-value method (assumes lock is already held)
             */
            inline double get_q_value_internal(const StateType &state, const ActionType &action) const {
                if (use_double_q_learning_) {
                    double q_a = get_q_value_from_table(q_table_a_, state, action);
                    double q_b = get_q_value_from_table(q_table_b_, state, action);
                    return (q_a + q_b) / 2.0;
                } else {
                    return get_q_value_from_table(q_table_a_, state, action);
                }
            }

            inline void update_single_q(const StateType &state, const ActionType &action, double reward,
                                        const StateType &next_state, bool terminal) {
                double current_q = get_q_value_from_table(q_table_a_, state, action);
                double max_next_q = terminal ? 0.0 : get_max_q_value_from_table(q_table_a_, next_state);
                double target = reward + gamma_ * max_next_q;
                double td_error = target - current_q;

                q_table_a_[state][action] = current_q + alpha_ * td_error;

                // Update average Q-value for statistics
                update_average_q_value();
            }

            inline void update_double_q(const StateType &state, const ActionType &action, double reward,
                                        const StateType &next_state, bool terminal) {
                // Randomly choose which Q-table to update
                bool update_a = uniform_dist_(rng_) < 0.5;

                if (update_a) {
                    // Update Q_A using Q_B for target
                    double current_q = get_q_value_from_table(q_table_a_, state, action);
                    ActionType best_next_action = get_best_action_from_table(q_table_a_, next_state);
                    double next_q = terminal ? 0.0 : get_q_value_from_table(q_table_b_, next_state, best_next_action);
                    double target = reward + gamma_ * next_q;

                    q_table_a_[state][action] = current_q + alpha_ * (target - current_q);
                } else {
                    // Update Q_B using Q_A for target
                    double current_q = get_q_value_from_table(q_table_b_, state, action);
                    ActionType best_next_action = get_best_action_from_table(q_table_b_, next_state);
                    double next_q = terminal ? 0.0 : get_q_value_from_table(q_table_a_, next_state, best_next_action);
                    double target = reward + gamma_ * next_q;

                    q_table_b_[state][action] = current_q + alpha_ * (target - current_q);
                }

                update_average_q_value();
            }

            inline void update_eligibility_traces(const StateType &state, const ActionType &action, double reward,
                                                  const StateType &next_state, bool terminal) {
                // Decay all eligibility traces
                for (auto &state_pair : eligibility_traces_) {
                    for (auto &action_pair : state_pair.second) {
                        action_pair.second *= gamma_ * lambda_;
                    }
                }

                // Set current state-action eligibility to 1
                eligibility_traces_[state][action] = 1.0;

                // Calculate TD error
                double current_q = get_q_value_from_table(q_table_a_, state, action);
                double max_next_q = terminal ? 0.0 : get_max_q_value_from_table(q_table_a_, next_state);
                double td_error = reward + gamma_ * max_next_q - current_q;

                // Update all Q-values proportional to eligibility traces
                for (const auto &state_pair : eligibility_traces_) {
                    for (const auto &action_pair : state_pair.second) {
                        if (action_pair.second > 1e-10) { // Only update if trace is significant
                            q_table_a_[state_pair.first][action_pair.first] += alpha_ * td_error * action_pair.second;
                        }
                    }
                }
            }

            inline void store_experience(const StateType &state, const ActionType &action, double reward,
                                         const StateType &next_state, bool terminal) {
                if (replay_buffer_.size() >= replay_buffer_capacity_) {
                    // Remove oldest experience
                    replay_buffer_.erase(replay_buffer_.begin());
                }

                replay_buffer_.emplace_back(state, action, reward, next_state, terminal);
            }

            inline ActionType epsilon_greedy_selection(const StateType &state,
                                                       const std::vector<ActionType> &valid_actions, bool &explored) {
                if (uniform_dist_(rng_) < epsilon_) {
                    explored = true;
                    std::uniform_int_distribution<size_t> dist(0, valid_actions.size() - 1);
                    return valid_actions[dist(rng_)];
                } else {
                    explored = false;
                    return get_best_action_from_actions(state, valid_actions);
                }
            }

            inline ActionType boltzmann_selection(const StateType &state, const std::vector<ActionType> &valid_actions,
                                                  bool &explored) {
                std::vector<double> probabilities;
                double max_q = std::numeric_limits<double>::lowest();

                // Find max Q-value for numerical stability
                for (const auto &action : valid_actions) {
                    max_q = std::max(max_q, get_q_value_internal(state, action));
                }

                // Calculate Boltzmann probabilities
                double sum = 0.0;
                for (const auto &action : valid_actions) {
                    double prob = std::exp((get_q_value_internal(state, action) - max_q) / temperature_);
                    probabilities.push_back(prob);
                    sum += prob;
                }

                // Normalize probabilities
                for (auto &prob : probabilities) {
                    prob /= sum;
                }

                // Sample action
                double rand_val = uniform_dist_(rng_);
                double cumsum = 0.0;
                for (size_t i = 0; i < valid_actions.size(); ++i) {
                    cumsum += probabilities[i];
                    if (rand_val <= cumsum) {
                        explored = (probabilities[i] < 0.9); // Heuristic for exploration detection
                        return valid_actions[i];
                    }
                }

                explored = false;
                return valid_actions.back();
            }

            inline ActionType ucb_selection(const StateType &state, const std::vector<ActionType> &valid_actions,
                                            bool &explored) {
                double best_ucb = std::numeric_limits<double>::lowest();
                ActionType best_action = valid_actions[0];
                size_t total_visits = state_visit_counts_[state];

                for (const auto &action : valid_actions) {
                    double q_value = get_q_value_internal(state, action);
                    size_t action_visits = visit_counts_[state][action];

                    double ucb_value = q_value;
                    if (action_visits > 0 && total_visits > 0) {
                        ucb_value += ucb_c_ * std::sqrt(std::log(total_visits) / action_visits);
                    } else {
                        ucb_value = std::numeric_limits<double>::max(); // Unvisited actions get highest priority
                    }

                    if (ucb_value > best_ucb) {
                        best_ucb = ucb_value;
                        best_action = action;
                    }
                }

                explored = (visit_counts_[state][best_action] < total_visits * 0.1); // Heuristic
                return best_action;
            }

            inline ActionType epsilon_decay_selection(const StateType &state,
                                                      const std::vector<ActionType> &valid_actions, bool &explored) {
                double current_epsilon =
                    std::max(min_epsilon_, epsilon_ * std::pow(epsilon_decay_, stats_.total_actions));

                if (uniform_dist_(rng_) < current_epsilon) {
                    explored = true;
                    std::uniform_int_distribution<size_t> dist(0, valid_actions.size() - 1);
                    return valid_actions[dist(rng_)];
                } else {
                    explored = false;
                    return get_best_action_from_actions(state, valid_actions);
                }
            }

            inline ActionType get_best_action_from_actions(const StateType &state,
                                                           const std::vector<ActionType> &actions) const {
                ActionType best_action = actions[0];
                double best_q = get_q_value_internal(state, best_action);

                for (const auto &action : actions) {
                    double q = get_q_value_internal(state, action);
                    if (q > best_q) {
                        best_q = q;
                        best_action = action;
                    }
                }

                return best_action;
            }

            inline ActionType get_best_action_from_table(
                const std::unordered_map<StateType, std::unordered_map<ActionType, double>> &table,
                const StateType &state) const {

                auto state_it = table.find(state);
                if (state_it == table.end() || state_it->second.empty()) {
                    if (available_actions_.empty()) {
                        throw std::runtime_error("No available actions.");
                    }
                    return available_actions_[action_dist_(rng_)];
                }

                auto best_action = std::max_element(state_it->second.begin(), state_it->second.end(),
                                                    [](const auto &a, const auto &b) { return a.second < b.second; });

                return best_action->first;
            }

            inline double
            get_q_value_from_table(const std::unordered_map<StateType, std::unordered_map<ActionType, double>> &table,
                                   const StateType &state, const ActionType &action) const {

                auto state_it = table.find(state);
                if (state_it == table.end()) {
                    return 0.0;
                }

                auto action_it = state_it->second.find(action);
                if (action_it == state_it->second.end()) {
                    return 0.0;
                }

                return action_it->second;
            }

            inline double get_max_q_value_from_table(
                const std::unordered_map<StateType, std::unordered_map<ActionType, double>> &table,
                const StateType &state) const {

                auto state_it = table.find(state);
                if (state_it == table.end() || state_it->second.empty()) {
                    return 0.0;
                }

                double max_q = std::numeric_limits<double>::lowest();
                for (const auto &action_value : state_it->second) {
                    max_q = std::max(max_q, action_value.second);
                }
                return max_q;
            }

            inline void update_learning_rate() {
                switch (lr_schedule_) {
                case LearningRateSchedule::LINEAR_DECAY:
                    alpha_ = std::max(min_alpha_, alpha_ - alpha_decay_ / stats_.total_updates);
                    break;
                case LearningRateSchedule::EXPONENTIAL_DECAY:
                    alpha_ = std::max(min_alpha_, alpha_ * alpha_decay_);
                    break;
                case LearningRateSchedule::ADAPTIVE:
                    // Implement adaptive learning rate based on performance
                    // This is a simplified version
                    if (stats_.total_updates > 1000 && stats_.total_updates % 1000 == 0) {
                        alpha_ *= 0.99; // Slowly decay
                        alpha_ = std::max(min_alpha_, alpha_);
                    }
                    break;
                case LearningRateSchedule::CONSTANT:
                default:
                    // Do nothing
                    break;
                }
            }

            inline void update_average_q_value() {
                if (q_table_a_.empty()) {
                    stats_.average_q_value = 0.0;
                    return;
                }

                double sum = 0.0;
                size_t count = 0;

                for (const auto &state_pair : q_table_a_) {
                    for (const auto &action_pair : state_pair.second) {
                        sum += action_pair.second;
                        count++;
                    }
                }

                stats_.average_q_value = count > 0 ? sum / count : 0.0;
            }
        };

    } // namespace model_free_value_based
} // namespace relearn
