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
    namespace common {

        /**
         * @brief Thompson Sampling Exploration
         *
         * Bayesian exploration strategy that samples from posterior
         * distributions of action values.
         *
         * Features to implement:
         * - Bayesian posterior sampling
         * - Beta/Gaussian distributions
         * - Uncertainty-based exploration
         * - Multi-armed bandit optimization
         */
        template <typename StateType, typename ActionType> class ThompsonSampling {
          public:
            // TODO: Implement Thompson Sampling
            // - Constructor with prior parameters
            // - update() method with posterior updates
            // - select_action() method with sampling
            // - Distribution parameter management

          private:
            // TODO: Internal state for Thompson Sampling
        };

        /**
         * @brief Information Gain Exploration
         *
         * Exploration strategy based on maximizing information gain
         * about the environment or value function.
         *
         * Features to implement:
         * - Information gain computation
         * - Uncertainty estimation
         * - Active learning principles
         * - Exploration-exploitation balance
         */
        template <typename StateType, typename ActionType> class InformationGainExploration {
          public:
            // TODO: Implement Information Gain Exploration
            // - Constructor with information metrics
            // - update() method with uncertainty updates
            // - select_action() method with information maximization
            // - Uncertainty quantification

          private:
            // TODO: Internal state for Information Gain Exploration
        };

    } // namespace common
} // namespace relearn
