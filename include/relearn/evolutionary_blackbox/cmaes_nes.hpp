#pragma once

#include <algorithm>
#include <random>
#include <vector>

namespace relearn {
    namespace evolutionary_blackbox {

        /**
         * @brief Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implementation
         *
         * Evolution strategy for black-box optimization of policy parameters.
         * Adapts covariance matrix based on successful mutations.
         */
        template <typename PolicyType> class CMAES {
          private:
            // Population parameters
            // Covariance matrix and evolution paths
            int population_size_;
            int dimension_;
            std::vector<double> mean_;
            std::vector<std::vector<double>> covariance_matrix_;

          public:
            inline CMAES(int dimension, int population_size = 0) : dimension_(dimension) {
                if (population_size == 0) {
                    population_size_ = 4 + static_cast<int>(3 * std::log(dimension));
                } else {
                    population_size_ = population_size;
                }
            }

            // Method signatures to be implemented
            inline std::vector<std::vector<double>> sample_population();
            inline void update_distribution(const std::vector<std::vector<double>> &population,
                                            const std::vector<double> &fitness_values);
            inline void adapt_covariance_matrix();
            inline std::vector<double> get_best_parameters() const;
            inline void set_initial_mean(const std::vector<double> &initial_mean);
        };

        /**
         * @brief Natural Evolution Strategies (NES) implementation
         *
         * Uses natural gradients to update search distribution.
         * More principled approach to evolution strategies.
         */
        template <typename PolicyType> class NES {
          private:
            // Search distribution parameters
            int population_size_;
            std::vector<double> theta_; // Distribution parameters
            double learning_rate_ = 0.01;

          public:
            inline NES(int parameter_dim, int population_size = 100)
                : population_size_(population_size), theta_(parameter_dim, 0.0) {}

            // Method signatures to be implemented
            inline std::vector<std::vector<double>> sample_population();
            inline void update_parameters(const std::vector<std::vector<double>> &population,
                                          const std::vector<double> &fitness_values);
            inline std::vector<double> compute_natural_gradient(const std::vector<std::vector<double>> &population,
                                                                const std::vector<double> &fitness_values);
            inline std::vector<double> get_best_parameters() const;
        };

    } // namespace evolutionary_blackbox
} // namespace relearn
