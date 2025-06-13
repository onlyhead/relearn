#pragma once

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

namespace relearn {
    namespace common {

        /**
         * @brief Base class for all reinforcement learning environments
         */
        template <typename StateType, typename ActionType> class Environment {
          public:
            virtual ~Environment() = default;

            // Pure virtual methods to be implemented by concrete environments
            virtual StateType reset() = 0;
            virtual std::tuple<StateType, double, bool> step(const ActionType &action) = 0;
            virtual std::vector<ActionType> get_action_space() const = 0;
            virtual bool is_terminal(const StateType &state) const = 0;
        };

        /**
         * @brief Experience replay buffer for off-policy algorithms
         */
        template <typename StateType, typename ActionType> class ReplayBuffer {
          private:
            struct Transition {
                StateType state;
                ActionType action;
                double reward;
                StateType next_state;
                bool done;
            };

            std::vector<Transition> buffer_;
            size_t capacity_;
            size_t position_;
            std::mt19937 rng_;

          public:
            inline ReplayBuffer(size_t capacity) : capacity_(capacity), position_(0), rng_(std::random_device{}()) {
                buffer_.reserve(capacity_);
            }

            // Method signatures to be implemented
            inline void add(const StateType &state, const ActionType &action, double reward,
                            const StateType &next_state, bool done) {
                Transition trans{state, action, reward, next_state, done};
                if (buffer_.size() < capacity_) {
                    buffer_.push_back(trans);
                } else {
                    buffer_[position_] = trans;
                }
                position_ = (position_ + 1) % capacity_;
            }

            inline std::vector<Transition> sample(size_t batch_size) {
                std::vector<Transition> batch;
                std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);
                for (size_t i = 0; i < batch_size; ++i) {
                    batch.push_back(buffer_[dist(rng_)]);
                }
                return batch;
            }

            inline size_t size() const { return buffer_.size(); }

            inline bool can_sample(size_t batch_size) const { return buffer_.size() >= batch_size; }
        };

        /**
         * @brief Utility functions for RL algorithms
         */
        class Utils {
          public: // Statistical functions
            inline static double compute_gae(const std::vector<double> &rewards, const std::vector<double> &values,
                                             double gamma = 0.99, double lambda = 0.95) {
                (void)rewards;
                (void)values;
                (void)gamma;
                (void)lambda;
                return 0.0; // Placeholder implementation
            }

            inline static std::vector<double> compute_returns(const std::vector<double> &rewards, double gamma = 0.99) {
                std::vector<double> returns(rewards.size());
                double running_return = 0.0;
                for (int i = rewards.size() - 1; i >= 0; --i) {
                    running_return = rewards[i] + gamma * running_return;
                    returns[i] = running_return;
                }
                return returns;
            }

            inline static std::vector<double> normalize(const std::vector<double> &values) {
                if (values.empty())
                    return values;
                double sum = 0.0, sum_sq = 0.0;
                for (double v : values) {
                    sum += v;
                    sum_sq += v * v;
                }
                double mean = sum / values.size();
                double variance = sum_sq / values.size() - mean * mean;
                double std_dev = std::sqrt(variance + 1e-8);

                std::vector<double> normalized(values.size());
                for (size_t i = 0; i < values.size(); ++i) {
                    normalized[i] = (values[i] - mean) / std_dev;
                }
                return normalized;
            }

            // Neural network utilities
            inline static std::vector<double> softmax(const std::vector<double> &logits) {
                if (logits.empty())
                    return logits;
                double max_logit = *std::max_element(logits.begin(), logits.end());
                std::vector<double> result(logits.size());
                double sum = 0.0;
                for (size_t i = 0; i < logits.size(); ++i) {
                    result[i] = std::exp(logits[i] - max_logit);
                    sum += result[i];
                }
                for (size_t i = 0; i < result.size(); ++i) {
                    result[i] /= sum;
                }
                return result;
            }

            inline static double log_sum_exp(const std::vector<double> &values) {
                if (values.empty())
                    return 0.0;
                double max_val = *std::max_element(values.begin(), values.end());
                double sum = 0.0;
                for (double v : values) {
                    sum += std::exp(v - max_val);
                }
                return max_val + std::log(sum);
            }
        };

    } // namespace common
} // namespace relearn
