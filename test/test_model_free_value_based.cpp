#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>
#include <vector>

using namespace relearn;

TEST_CASE("Q-Learning basic functionality") {
    // Actions: 0=up, 1=down, 2=left, 3=right
    std::vector<int> actions = {0, 1, 2, 3};
    model_free_value_based::QLearning<int, int> q_learning(0.1, 0.9, 0.1, actions);

    SUBCASE("Construction and parameter setting") {
        CHECK(q_learning.get_learning_rate() == doctest::Approx(0.1));
        CHECK(q_learning.get_discount_factor() == doctest::Approx(0.9));
        CHECK(q_learning.get_epsilon() == doctest::Approx(0.1));
        CHECK(q_learning.get_q_table_size() == 0);
    }

    SUBCASE("Initial Q-values are zero") {
        CHECK(q_learning.get_q_value(0, 0) == doctest::Approx(0.0));
        CHECK(q_learning.get_q_value(1, 2) == doctest::Approx(0.0));
        CHECK(q_learning.get_q_value(5, 3) == doctest::Approx(0.0));
    }

    SUBCASE("Q-learning update") {
        // Initial state 0, action 1, reward 10, next state 1
        q_learning.update(0, 1, 10.0, 1);

        // Q(0,1) should be updated: Q(0,1) = 0 + 0.1 * (10 + 0.9 * 0 - 0) = 1.0
        CHECK(q_learning.get_q_value(0, 1) == doctest::Approx(1.0));
        CHECK(q_learning.get_q_table_size() == 1);

        // Add another Q-value
        q_learning.update(1, 2, 5.0, 2);
        CHECK(q_learning.get_q_value(1, 2) == doctest::Approx(0.5));
        CHECK(q_learning.get_q_table_size() == 2);

        // Update existing Q-value with dependency
        q_learning.update(0, 1, 2.0, 1);
        // Q(0,1) = 1.0 + 0.1 * (2.0 + 0.9 * 0.5 - 1.0) = 1.0 + 0.1 * 1.45 = 1.145
        CHECK(q_learning.get_q_value(0, 1) == doctest::Approx(1.145));
    }

    SUBCASE("Action selection") {
        // Set a deterministic scenario for testing
        q_learning.set_epsilon(0.0); // No exploration, always greedy

        // Set some Q-values
        q_learning.update(0, 0, 1.0, 1);
        q_learning.update(0, 1, 5.0, 1); // Best action
        q_learning.update(0, 2, 2.0, 1);

        // Should select action 1 (highest Q-value)
        int selected_action = q_learning.get_best_action(0);
        CHECK(selected_action == 1);
    }

    SUBCASE("Max Q-value calculation") {
        q_learning.update(5, 0, 3.0, 6);
        q_learning.update(5, 1, 7.0, 6);
        q_learning.update(5, 2, 1.0, 6);

        CHECK(q_learning.get_max_q_value(5) == doctest::Approx(0.7));
        CHECK(q_learning.get_max_q_value(999) == doctest::Approx(0.0)); // Unknown state
    }

    SUBCASE("Parameter modification") {
        q_learning.set_learning_rate(0.5);
        q_learning.set_discount_factor(0.8);
        q_learning.set_epsilon(0.2);

        CHECK(q_learning.get_learning_rate() == doctest::Approx(0.5));
        CHECK(q_learning.get_discount_factor() == doctest::Approx(0.8));
        CHECK(q_learning.get_epsilon() == doctest::Approx(0.2));
    }
}

TEST_CASE("Q-Learning simple grid world scenario") {
    // Simple 2x2 grid world: states 0,1,2,3 and actions 0=up,1=down,2=left,3=right
    std::vector<int> actions = {0, 1, 2, 3};
    model_free_value_based::QLearning<int, int> agent(0.1, 0.9, 0.0, actions); // No exploration

    // Train on a simple episode: 0 -> 1 -> 3 (goal)
    // State 3 gives reward 100, others give 0
    agent.update(0, 3, 0.0, 1);   // Move right from 0 to 1
    agent.update(1, 1, 100.0, 3); // Move down from 1 to 3 (goal, high reward)

    // After some updates, agent should prefer the path that leads to high reward
    CHECK(agent.get_q_value(1, 1) > agent.get_q_value(1, 0));
    CHECK(agent.get_q_value(1, 1) > agent.get_q_value(1, 2));
}

TEST_CASE("DQN basic functionality") {
    model_free_value_based::DQN<std::vector<double>, int> dqn;

    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}

TEST_CASE("Common utilities") {
    SUBCASE("ReplayBuffer construction") {
        common::ReplayBuffer<int, int> buffer(1000);
        CHECK(buffer.size() == 0);
    }
}
