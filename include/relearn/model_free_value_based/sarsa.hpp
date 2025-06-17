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
         * On-policy temporal difference learning algorithm.
         * Unlike Q-learning, SARSA updates the policy being followed.
         *
         * Features to implement:
         * - On-policy learning
         * - Eligibility traces (SARSA(λ))
         * - Multiple exploration strategies
         * - Experience replay adaptation
         * - Performance monitoring
         * - Thread-safe operations
         */
        template <typename StateType, typename ActionType> class SARSA {
          public:
            // TODO: Implement SARSA algorithm
            // - Constructor with learning rate, discount, exploration
            // - update() method for on-policy learning
            // - select_action() with current policy
            // - Eligibility traces support
            // - Integration with existing exploration strategies

          private:
            // TODO: Internal state for SARSA
        };

    } // namespace model_free_value_based
} // namespace relearn
