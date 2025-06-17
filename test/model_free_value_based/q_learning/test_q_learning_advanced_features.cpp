#include <doctest/doctest.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <relearn/relearn.hpp>
#include <set>
#include <thread>
#include <vector>

using namespace relearn::model_free_value_based;

TEST_CASE("Q-Learning Advanced Features - Exploration Strategies") {
    std::vector<int> actions = {0, 1, 2, 3};

    SUBCASE("Epsilon-greedy exploration with high epsilon") {
        QLearning<int, int> agent(0.1, 0.9, 0.9, actions); // High epsilon for exploration
        
        // Test multiple action selections to verify exploration
        std::set<int> selected_actions;
        for (int i = 0; i < 100; ++i) {
            int action = agent.select_action(0);
            selected_actions.insert(action);
        }
        
        // With high epsilon, we should see exploration (multiple actions)
        CHECK(selected_actions.size() > 1);
    }

    SUBCASE("Low epsilon should be more greedy") {
        QLearning<int, int> agent(0.1, 0.9, 0.01, actions); // Low epsilon
        
        // Train one action to be clearly better
        for (int i = 0; i < 100; ++i) {
            agent.update(0, 1, 10.0, 1, false); // Make action 1 very good
        }
        
        // Now most selections should prefer action 1
        int action_1_count = 0;
        for (int i = 0; i < 100; ++i) {
            if (agent.select_action(0) == 1) {
                action_1_count++;
            }
        }
        
        // Should select action 1 most of the time with low epsilon
        CHECK(action_1_count > 80);
    }
}

TEST_CASE("Q-Learning Advanced Features - Double Q-Learning") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Double Q-learning can be enabled and disabled") {
        agent.set_double_q_learning(true);
        CHECK_NOTHROW(agent.update(0, 1, 1.0, 1, false));
        
        agent.set_double_q_learning(false);
        CHECK_NOTHROW(agent.update(0, 1, 1.0, 1, false));
    }
}

TEST_CASE("Q-Learning Advanced Features - Eligibility Traces") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Eligibility traces can be enabled") {
        agent.set_eligibility_traces(true, 0.9);
        
        // Perform sequence of updates
        agent.update(0, 1, 1.0, 1, false);
        agent.update(1, 2, 1.0, 2, false);
        agent.update(2, 3, 1.0, 3, true);
        
        // Q-values should propagate backwards due to eligibility traces
        double q_0_1 = agent.get_q_value(0, 1);
        CHECK(q_0_1 > 0.0);
    }
}

TEST_CASE("Q-Learning Advanced Features - Experience Replay") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Experience replay can be enabled") {
        agent.set_experience_replay(true, 1000, 32);
        
        // Add some experiences
        for (int i = 0; i < 50; ++i) {
            agent.update(i % 4, (i + 1) % 4, 1.0, (i + 1) % 4, false);
        }
        
        // Should work without throwing
        CHECK_NOTHROW(agent.replay_experience());
    }
}

TEST_CASE("Q-Learning Advanced Features - Learning Rate Adjustment") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.5, 0.9, 0.1, actions);
    
    SUBCASE("Learning rate can be modified") {
        double initial_lr = agent.get_learning_rate();
        CHECK(initial_lr == doctest::Approx(0.5));
        
        agent.set_learning_rate(0.1);
        double new_lr = agent.get_learning_rate();
        CHECK(new_lr == doctest::Approx(0.1));
        CHECK(new_lr < initial_lr);
    }
}

TEST_CASE("Q-Learning Advanced Features - Action Masking") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions); // No exploration for deterministic testing
    
    SUBCASE("Action masking restricts available actions") {
        // Mask that only allows actions 0 and 2
        std::function<bool(const int&, const int&)> mask = [](const int& state, const int& action) {
            return action == 0 || action == 2;
        };
        agent.set_action_mask(mask);
        
        // Test multiple selections
        std::set<int> selected_actions;
        for (int i = 0; i < 20; ++i) {
            int action = agent.select_action(0);
            selected_actions.insert(action);
        }
        
        // Should only contain actions 0 and 2
        for (int action : selected_actions) {
            CHECK((action == 0 || action == 2));
        }
    }
}

TEST_CASE("Q-Learning Advanced Features - Reward Shaping") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Reward shaping modifies rewards") {
        // Shaping function that doubles the reward
        std::function<double(double)> shaping = [](double reward) {
            return reward * 2.0;
        };
        agent.set_reward_shaping(shaping);
        
        // The actual reward processing happens internally
        // We can only test that it doesn't throw
        CHECK_NOTHROW(agent.update(0, 1, 1.0, 1, false));
    }
}

TEST_CASE("Q-Learning Advanced Features - Performance Monitoring") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Statistics track performance") {
        auto initial_stats = agent.get_statistics();
        CHECK(initial_stats.total_updates == 0);
        CHECK(initial_stats.total_actions == 0);
        
        // Perform some updates
        for (int i = 0; i < 10; ++i) {
            agent.select_action(0); // This should increment total_actions
            agent.update(0, 1, 1.0, 1, false); // This should increment total_updates
        }
        
        auto final_stats = agent.get_statistics();
        CHECK(final_stats.total_updates == 10);
        CHECK(final_stats.total_actions == 10);
    }
}

TEST_CASE("Q-Learning Advanced Features - Thread Safety") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Concurrent updates should not crash") {
        std::vector<std::thread> threads;
        
        // Launch multiple threads doing updates
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&agent, t]() {
                for (int i = 0; i < 100; ++i) {
                    agent.update(t, (t + 1) % 4, 1.0, (t + 1) % 4, false);
                }
            });
        }
        
        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Should have completed without crashing
        auto stats = agent.get_statistics();
        CHECK(stats.total_updates == 400);
    }
}

TEST_CASE("Q-Learning Advanced Features - Q-Table Persistence") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Q-table can be saved and loaded") {
        // Train the agent a bit
        for (int i = 0; i < 50; ++i) {
            agent.update(i % 4, (i + 1) % 4, 1.0, (i + 1) % 4, false);
        }
        
        double trained_q_value = agent.get_q_value(0, 1);
        
        // Save and load
        std::string filename = "test_q_table.bin";
        CHECK_NOTHROW(agent.save_q_table(filename));
        
        // Create new agent and load
        QLearning<int, int> new_agent(0.1, 0.9, 0.1, actions);
        CHECK_NOTHROW(new_agent.load_q_table(filename));
        
        // Should have same Q-value
        double loaded_q_value = new_agent.get_q_value(0, 1);
        CHECK(loaded_q_value == doctest::Approx(trained_q_value));
    }
}
