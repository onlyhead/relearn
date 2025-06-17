#pragma once

/**
 * @file relearn.hpp
 * @brief Main header file for the ReLearn reinforcement learning library
 *
 * A comprehensive header-only C++ library implementing various reinforcement learning algorithms
 * organized by categories as outlined in the GUIDE.md
 */

// Common utilities and base classes
#include "common/base.hpp"

// Model-Free, Value-Based algorithms
#include "model_free_value_based/advanced_algorithms.hpp"
#include "model_free_value_based/dqn.hpp"
#include "model_free_value_based/dqn_variants.hpp"
#include "model_free_value_based/expected_sarsa.hpp"
#include "model_free_value_based/q_learning.hpp"
#include "model_free_value_based/sarsa.hpp"

// Model-Free, Policy-Gradient algorithms
#include "model_free_policy_gradient/ppo.hpp"
#include "model_free_policy_gradient/reinforce.hpp"
#include "model_free_policy_gradient/trpo.hpp"

// Model-Free, Actor-Critic algorithms
#include "model_free_actor_critic/a2c_a3c.hpp"
#include "model_free_actor_critic/ddpg.hpp"
#include "model_free_actor_critic/sac.hpp"
#include "model_free_actor_critic/td3.hpp"

// Model-Based RL algorithms
#include "model_based/dreamer_pets.hpp"
#include "model_based/mbpo.hpp"
#include "model_based/pilco.hpp"
#include "model_based/planning_algorithms.hpp"

// Imitation & Inverse RL algorithms
#include "imitation_inverse/behavioral_cloning.hpp"
#include "imitation_inverse/dagger.hpp"
#include "imitation_inverse/gail.hpp"

// Hierarchical & Meta-RL algorithms
#include "hierarchical_meta/maml_rl2.hpp"
#include "hierarchical_meta/options_feudal.hpp"

// Evolutionary & Black-Box Methods
#include "evolutionary_blackbox/cmaes_nes.hpp"

// Multi-Agent Reinforcement Learning
#include "multi_agent/cooperative_learning.hpp"
#include "multi_agent/joint_learning.hpp"

// Advanced Exploration Strategies
#include "common/advanced_exploration.hpp"

/**
 * @namespace relearn
 * @brief Main namespace for the ReLearn library
 */
namespace relearn {
    // Version information
    constexpr int VERSION_MAJOR = 1;
    constexpr int VERSION_MINOR = 0;
    constexpr int VERSION_PATCH = 0;
} // namespace relearn