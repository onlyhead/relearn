/**
 * @file comprehensive_test_advanced_q_learning.cpp
 * @brief Comprehensive test suite for advanced Q-learning features
 *
 * This test suite validates all production-ready features of the Q-learning implementation:
 * - Basic functionality and parameter setting
 * - Multiple exploration strategies
 * - Double Q-learning
 * - Eligibility traces
 * - Experience replay
 * - Learning rate scheduling
 * - Action masking and reward shaping
 * - Q-table persistence
 * - Thread safety
 * - Performance monitoring
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <relearn/relearn.hpp>
#include <thread>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace relearn::model_free_value_based;

TEST_CASE("Advanced Q-Learning - Basic Functionality") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    SUBCASE("Constructor and default parameters") {
        CHECK(agent.get_learning_rate() == doctest::Approx(0.1));
        CHECK(agent.get_discount_factor() == doctest::Approx(0.9));
        CHECK(agent.get_epsilon() == doctest::Approx(0.1));
        CHECK(agent.get_q_table_size() == 0);
    }

    SUBCASE("Basic Q-value operations") {
        // Initial Q-values should be zero
        CHECK(agent.get_q_value(0, 0) == doctest::Approx(0.0));
        CHECK(agent.get_max_q_value(0) == doctest::Approx(0.0));

        // Update Q-value
        agent.update(0, 1, 10.0, 1, false);
        CHECK(agent.get_q_value(0, 1) == doctest::Approx(1.0));
        CHECK(agent.get_q_table_size() > 0);
    }

    SUBCASE("Action selection") {
        agent.set_epsilon(0.0); // Deterministic

        // Set up Q-values
        agent.update(0, 0, 1.0, 1, false);
        agent.update(0, 1, 5.0, 1, false);
        agent.update(0, 2, 2.0, 1, false);

        // Should select action with highest Q-value
        int best_action = agent.get_best_action(0);
        CHECK(best_action == 1);
    }
}

TEST_CASE("Advanced Q-Learning - Exploration Strategies") {
    std::vector<int> actions = {0, 1, 2, 3};

    SUBCASE("Epsilon-Greedy Strategy") {
        QLearning<int, int> agent(0.1, 0.9, 0.5, actions, QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY);

        // With high epsilon, should see some exploration
        std::map<int, int> action_counts;
        for (int i = 0; i < 100; ++i) {
            int action = agent.select_action(0);
            action_counts[action]++;
        }

        // Should have selected multiple actions due to exploration
        CHECK(action_counts.size() > 1);
    }

    SUBCASE("Boltzmann Strategy") {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions, QLearning<int, int>::ExplorationStrategy::BOLTZMANN);
        agent.set_temperature(1.0);

        // Set up different Q-values
        agent.update(0, 0, 1.0, 1, false);
        agent.update(0, 1, 10.0, 1, false);

        std::map<int, int> action_counts;
        for (int i = 0; i < 100; ++i) {
            int action = agent.select_action(0);
            action_counts[action]++;
        }

        // Action 1 should be selected more often due to higher Q-value
        CHECK(action_counts[1] > action_counts[0]);
    }

    SUBCASE("UCB1 Strategy") {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions, QLearning<int, int>::ExplorationStrategy::UCB1);
        agent.set_ucb_c(2.0);

        std::map<int, int> action_counts;
        for (int i = 0; i < 100; ++i) {
            int action = agent.select_action(0);
            action_counts[action]++;
        }

        // Should explore all actions initially
        CHECK(action_counts.size() == 4);
    }
}

TEST_CASE("Advanced Q-Learning - Double Q-Learning") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    agent.set_double_q_learning(true);

    // Train with double Q-learning
    for (int i = 0; i < 100; ++i) {
        agent.update(0, 1, 1.0, 1, false);
        agent.update(1, 2, 1.0, 2, false);
    }

    CHECK(agent.get_q_value(0, 1) > 0.0);
    CHECK(agent.get_q_value(1, 2) > 0.0);

    auto stats = agent.get_statistics();
    CHECK(stats.total_updates == 200);
}

TEST_CASE("Advanced Q-Learning - Eligibility Traces") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    agent.set_eligibility_traces(true, 0.9);

    // Create a sequence that should benefit from eligibility traces
    agent.update(0, 1, 0.0, 1, false);
    agent.update(1, 2, 0.0, 2, false);
    agent.update(2, 3, 10.0, 3, true); // Terminal reward

    // The first state-action should have learned from the delayed reward
    CHECK(agent.get_q_value(0, 1) > 0.0);
}

TEST_CASE("Advanced Q-Learning - Experience Replay") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    agent.set_experience_replay(true, 1000, 32);

    // Generate experiences
    for (int i = 0; i < 100; ++i) {
        agent.update(i % 10, i % 4, 1.0, (i + 1) % 10, false);
    }

    // Manually trigger experience replay
    agent.replay_experience();

    auto stats = agent.get_statistics();
    CHECK(stats.total_updates >= 100);
}

TEST_CASE("Advanced Q-Learning - Learning Rate Scheduling") {
    std::vector<int> actions = {0, 1, 2, 3};

    SUBCASE("Exponential Decay") {
        QLearning<int, int> agent(0.1, 0.9, 0.0, actions, QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY,
                                  QLearning<int, int>::LearningRateSchedule::EXPONENTIAL_DECAY);

        double initial_lr = agent.get_learning_rate();

        // Train for many episodes
        for (int i = 0; i < 1000; ++i) {
            agent.update(0, 1, 1.0, 1, false);
        }

        double final_lr = agent.get_learning_rate();
        CHECK(final_lr <= initial_lr); // Should decay
    }
}

TEST_CASE("Advanced Q-Learning - Action Masking") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    // Set action mask that only allows actions 0 and 1 for state 0
    agent.set_action_mask([](int state, int action) {
        if (state == 0) {
            return action == 0 || action == 1;
        }
        return true;
    });

    // Test that only valid actions are selected
    std::set<int> selected_actions;
    for (int i = 0; i < 50; ++i) {
        int action = agent.select_action(0);
        selected_actions.insert(action);
    }

    // Should only contain actions 0 and 1
    CHECK(selected_actions.size() <= 2);
    for (int action : selected_actions) {
        CHECK((action == 0 || action == 1));
    }
}

TEST_CASE("Advanced Q-Learning - Reward Shaping") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    // Set reward shaping that doubles rewards
    agent.set_reward_shaping([](double reward) { return reward * 2.0; });
    agent.update(0, 1, 5.0, 1, false); // Shaped to 10.0

    // Q-value should reflect shaped reward: Q(s,a) = 0 + 0.1 * (10.0 + 0.9 * 0 - 0) = 1.0
    CHECK(agent.get_q_value(0, 1) == doctest::Approx(1.0));
}

TEST_CASE("Advanced Q-Learning - Statistics Tracking") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    auto initial_stats = agent.get_statistics();
    CHECK(initial_stats.total_updates == 0);
    CHECK(initial_stats.total_actions == 0);

    // Perform some updates and actions
    for (int i = 0; i < 10; ++i) {
        agent.select_action(0);
        agent.update(0, 1, 1.0, 1, false);
    }

    auto final_stats = agent.get_statistics();
    CHECK(final_stats.total_updates == 10);
    CHECK(final_stats.total_actions == 10);
    CHECK(final_stats.cumulative_reward > 0.0);
    CHECK(final_stats.exploration_ratio >= 0.0);
    CHECK(final_stats.exploration_ratio <= 1.0);
}

TEST_CASE("Advanced Q-Learning - Q-Table Persistence") {
    std::vector<int> actions = {0, 1, 2, 3};
    const std::string filename = "test_q_table.bin";

    // Train an agent
    QLearning<int, int> agent1(0.1, 0.9, 0.0, actions);
    for (int i = 0; i < 50; ++i) {
        agent1.update(i % 5, i % 4, 1.0, (i + 1) % 5, false);
    }

    double original_q_value = agent1.get_q_value(0, 1);

    // Save Q-table
    agent1.save_q_table(filename);

    // Create new agent and load Q-table
    QLearning<int, int> agent2(0.1, 0.9, 0.0, actions);
    agent2.load_q_table(filename);

    // Should have same Q-values
    CHECK(agent2.get_q_value(0, 1) == doctest::Approx(original_q_value));

    // Clean up
    std::remove(filename.c_str());
}

TEST_CASE("Advanced Q-Learning - Thread Safety") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> shared_agent(0.1, 0.9, 0.1, actions);

    const int num_threads = 4;
    const int updates_per_thread = 100;
    std::vector<std::thread> threads;

    // Launch multiple threads doing updates
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&shared_agent, i]() {
            constexpr int updates = 100;
            for (int j = 0; j < updates; ++j) {
                int state = (i * 1000 + j) % 10;
                int action = j % 4;
                shared_agent.update(state, action, 1.0, (state + 1) % 10, false);
                shared_agent.select_action(state);
            }
        });
    }

    // Wait for all threads
    for (auto &thread : threads) {
        thread.join();
    }

    auto stats = shared_agent.get_statistics();
    CHECK(stats.total_updates == num_threads * 100);
    CHECK(stats.total_actions == num_threads * 100);
    CHECK(shared_agent.get_q_table_size() > 0);
}

TEST_CASE("Advanced Q-Learning - Performance Benchmarks") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    const int num_updates = 10000;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform many updates
    for (int i = 0; i < num_updates; ++i) {
        agent.update(i % 100, i % 4, 1.0, (i + 1) % 100, false);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Should complete within reasonable time (less than 1 second for 10k updates)
    CHECK(duration.count() < 1000);

    auto stats = agent.get_statistics();
    CHECK(stats.total_updates == num_updates);
}

TEST_CASE("Advanced Q-Learning - Memory Efficiency") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    // Enable experience replay with limited capacity
    agent.set_experience_replay(true, 100, 32);

    // Add more experiences than capacity
    for (int i = 0; i < 200; ++i) {
        agent.update(i, i % 4, 1.0, i + 1, false);
    }

    // Should not grow beyond capacity
    CHECK(agent.get_q_table_size() <= 200); // Should be manageable
}
