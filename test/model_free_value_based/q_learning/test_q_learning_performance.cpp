#include <doctest/doctest.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <relearn/relearn.hpp>
#include <vector>

using namespace relearn::model_free_value_based;

TEST_CASE("Q-Learning Performance - Training Speed") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Large number of updates should complete quickly") {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform many updates
        for (int i = 0; i < 10000; ++i) {
            agent.update(i % 10, (i + 1) % 4, 1.0, (i + 1) % 10, false);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Should complete in reasonable time (less than 1 second)
        CHECK(duration.count() < 1000);
        
        // Verify statistics were updated
        auto stats = agent.get_statistics();
        CHECK(stats.total_updates == 10000);
    }
}

TEST_CASE("Q-Learning Performance - Memory Usage") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Q-table grows with state-action pairs") {
        // Initially should have minimal Q-values
        CHECK(agent.get_q_value(0, 0) == doctest::Approx(0.0));
        
        // Add many different state-action pairs
        for (int state = 0; state < 100; ++state) {
            for (int action : actions) {
                agent.update(state, action, 1.0, (state + 1) % 100, false);
            }
        }
        
        // Should be able to retrieve all Q-values
        bool all_updated = true;
        for (int state = 0; state < 100; ++state) {
            for (int action : actions) {
                if (agent.get_q_value(state, action) == 0.0) {
                    all_updated = false;
                    break;
                }
            }
            if (!all_updated) break;
        }
        
        CHECK(all_updated);
    }
}

TEST_CASE("Q-Learning Performance - Convergence") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Agent should learn simple pattern") {
        // Simple environment: state 0 -> action 1 -> state 1 -> reward 10
        //                    state 0 -> other actions -> state 0 -> reward -1
        
        // Train for many episodes
        for (int episode = 0; episode < 1000; ++episode) {
            int state = 0;
            int action = agent.select_action(state);
            
            if (action == 1) {
                agent.update(state, action, 10.0, 1, true); // Good action
            } else {
                agent.update(state, action, -1.0, 0, false); // Bad action
            }
        }
        
        // After training, action 1 should have highest Q-value
        double q_action_1 = agent.get_q_value(0, 1);
        bool action_1_is_best = true;
        
        for (int other_action : {0, 2, 3}) {
            if (agent.get_q_value(0, other_action) >= q_action_1) {
                action_1_is_best = false;
                break;
            }
        }
        
        CHECK(action_1_is_best);
        CHECK(q_action_1 > 0); // Should be positive due to rewards
    }
}

TEST_CASE("Q-Learning Performance - Experience Replay Efficiency") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Experience replay should not slow down training significantly") {
        agent.set_experience_replay(true, 1000, 32);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform updates with experience replay
        for (int i = 0; i < 1000; ++i) {
            agent.update(i % 10, (i + 1) % 4, 1.0, (i + 1) % 10, false);
            if (i % 10 == 0) {
                agent.replay_experience();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Should still complete in reasonable time
        CHECK(duration.count() < 2000); // Allow more time for replay
    }
}

TEST_CASE("Q-Learning Performance - Statistics Tracking") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
    
    SUBCASE("Statistics should be accurate and efficient") {
        const int num_updates = 1000;
        const int num_actions = 500;
        
        // Perform actions and updates
        for (int i = 0; i < num_actions; ++i) {
            agent.select_action(i % 10);
        }
        
        for (int i = 0; i < num_updates; ++i) {
            agent.update(i % 10, (i + 1) % 4, 1.0, (i + 1) % 10, false);
        }
        
        auto stats = agent.get_statistics();
        CHECK(stats.total_updates == num_updates);
        CHECK(stats.total_actions == num_actions);
        CHECK(stats.cumulative_reward > 0); // Should have accumulated positive rewards
    }
}
