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
         * @brief Expected SARSA Algorithm
         *
         * Combines benefits of Q-learning and SARSA by using expected value
         * of next action instead of the actual next action.
         *
         * Features to implement:
         * - Expected value computation
         * - Policy-aware updates
         * - Better convergence properties than SARSA
         * - Integration with exploration strategies
         * - Experience replay support
         * - Performance monitoring
         */
        template <typename StateType, typename ActionType> class ExpectedSARSA {
          public:
            // TODO: Implement Expected SARSA algorithm
            // - Constructor with learning rate, discount, exploration
            // - update() method with expected value computation
            // - select_action() method
            // - Integration with existing exploration strategies

          private:
            // TODO: Internal state for Expected SARSA
        };

    } // namespace model_free_value_based
} // namespace relearn
