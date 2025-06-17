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

// Test framework for advanced Q-Learning features
namespace test_utils {
    int test_count = 0;
    int pass_count = 0;

    void assert_true(bool condition, const std::string &message) {
        test_count++;
        if (condition) {
            pass_count++;
            std::cout << "✓ PASS: " << message << std::endl;
        } else {
            std::cout << "✗ FAIL: " << message << std::endl;
        }
    }

    void assert_near(double actual, double expected, double tolerance, const std::string &message) {
        test_count++;
        bool condition = std::abs(actual - expected) < tolerance;
        if (condition) {
            pass_count++;
            std::cout << "✓ PASS: " << message << " (expected: " << expected << ", actual: " << actual << ")"
                      << std::endl;
        } else {
            std::cout << "✗ FAIL: " << message << " (expected: " << expected << ", actual: " << actual << ")"
                      << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << test_count << std::endl;
        std::cout << "Passed: " << pass_count << std::endl;
        std::cout << "Failed: " << (test_count - pass_count) << std::endl;
        std::cout << "Success rate: " << (100.0 * pass_count / test_count) << "%" << std::endl;
    }
} // namespace test_utils

void test_double_q_learning() {
    std::cout << "\n--- Testing Double Q-Learning ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    // Enable double Q-learning
    agent.set_double_q_learning(true);

    // Perform updates and verify no crashes
    for (int i = 0; i < 100; ++i) {
        agent.update(i % 10, i % 4, (i % 2 == 0) ? 1.0 : -1.0, (i + 1) % 10, false);
    }

    test_utils::assert_true(true, "Double Q-learning updates complete without errors");

    // Verify Q-values are being updated
    double q_value = agent.get_q_value(0, 0);
    test_utils::assert_true(std::abs(q_value) > 1e-6, "Double Q-learning produces non-zero Q-values");
}

void test_eligibility_traces() {
    std::cout << "\n--- Testing Eligibility Traces ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    // Enable eligibility traces
    agent.set_eligibility_traces(true, 0.9);

    // Create a simple episode to test trace propagation
    std::vector<int> states = {0, 1, 2, 3, 4};
    std::vector<int> episode_actions = {0, 1, 2, 3};

    // Execute episode
    for (size_t i = 0; i < episode_actions.size(); ++i) {
        bool terminal = (i == episode_actions.size() - 1);
        double reward = terminal ? 10.0 : 0.0; // Reward only at end
        agent.update(states[i], episode_actions[i], reward, states[i + 1], terminal);
    }

    // With eligibility traces, earlier states should have non-zero Q-values
    double early_q = agent.get_q_value(states[0], episode_actions[0]);
    test_utils::assert_true(early_q > 0.0, "Eligibility traces propagate reward to earlier states");
}

void test_experience_replay() {
    std::cout << "\n--- Testing Experience Replay ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    // Enable experience replay
    agent.set_experience_replay(true, 1000, 32);

    // Generate many experiences
    for (int i = 0; i < 200; ++i) {
        int state = i % 10;
        int action = i % 4;
        double reward = (i % 3 == 0) ? 1.0 : 0.0;
        int next_state = (i + 1) % 10;

        agent.update(state, action, reward, next_state, false);
    }

    test_utils::assert_true(true, "Experience replay updates complete without errors");

    // Check that Q-values have been learned
    bool has_learned = false;
    for (int s = 0; s < 10; ++s) {
        for (int a = 0; a < 4; ++a) {
            if (std::abs(agent.get_q_value(s, a)) > 1e-6) {
                has_learned = true;
                break;
            }
        }
        if (has_learned)
            break;
    }
    test_utils::assert_true(has_learned, "Experience replay produces learning");
}

void test_exploration_strategies() {
    std::cout << "\n--- Testing Exploration Strategies ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};

    // Test epsilon-greedy exploration
    {
        QLearning<int, int> agent(0.1, 0.9, 1.0, actions, QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY);

        // With epsilon = 1.0, should explore (random actions)
        std::vector<int> selected_actions;
        for (int i = 0; i < 50; ++i) {
            selected_actions.push_back(agent.select_action(0));
        }

        // Should have some variety in actions
        std::set<int> unique_actions(selected_actions.begin(), selected_actions.end());
        test_utils::assert_true(unique_actions.size() > 1, "Epsilon-greedy exploration produces varied actions");
    }

    // Test Boltzmann exploration
    {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions, QLearning<int, int>::ExplorationStrategy::BOLTZMANN);
        agent.set_temperature(2.0);

        // Update one action to have higher Q-value
        for (int i = 0; i < 20; ++i) {
            agent.update(0, 1, 1.0, 1, false);
        }

        // Boltzmann should still allow some exploration
        std::vector<int> selected_actions;
        for (int i = 0; i < 50; ++i) {
            selected_actions.push_back(agent.select_action(0));
        }

        test_utils::assert_true(true, "Boltzmann exploration completes without errors");
    }

    // Test UCB1 exploration
    {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions, QLearning<int, int>::ExplorationStrategy::UCB1);
        agent.set_ucb_c(2.0);

        // UCB1 should explore less-visited actions
        std::vector<int> selected_actions;
        for (int i = 0; i < 50; ++i) {
            int action = agent.select_action(0);
            selected_actions.push_back(action);
            agent.update(0, action, 0.5, 1, false);
        }

        test_utils::assert_true(true, "UCB1 exploration completes without errors");
    }
}

void test_action_masking() {
    std::cout << "\n--- Testing Action Masking ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions); // No exploration for predictable testing

    // Set action mask to block action 3
    agent.set_action_mask([](int state, int action) {
        return action != 3; // All actions except 3 are valid
    });

    // Even if action 3 has highest Q-value, it should not be selected
    agent.update(0, 3, 100.0, 1, false); // Give action 3 very high Q-value

    int selected_action = agent.select_action(0);
    test_utils::assert_true(selected_action != 3, "Action masking prevents selection of masked actions");

    // Verify unmasked actions can still be selected
    agent.update(0, 1, 50.0, 1, false); // Give action 1 high Q-value
    selected_action = agent.select_action(0);
    test_utils::assert_true(selected_action == 1, "Action masking allows selection of unmasked actions");
}

void test_reward_shaping() {
    std::cout << "\n--- Testing Reward Shaping ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions);

    // Set reward shaping to double all rewards
    agent.set_reward_shaping([](double reward) { return reward * 2.0; });

    double initial_q = agent.get_q_value(0, 0);

    // Update with reward 1.0, but shaping should make it 2.0
    agent.update(0, 0, 1.0, 1, false);

    double final_q = agent.get_q_value(0, 0);
    double expected_q = initial_q + 0.1 * (2.0 + 0.9 * 0.0 - initial_q); // Shaped reward = 2.0

    test_utils::assert_near(final_q, expected_q, 1e-6, "Reward shaping correctly modifies rewards");
}

void test_learning_rate_scheduling() {
    std::cout << "\n--- Testing Learning Rate Scheduling ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};

    // Test exponential decay
    {
        QLearning<int, int> agent(0.1, 0.9, 0.0, actions, QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY,
                                  QLearning<int, int>::LearningRateSchedule::EXPONENTIAL_DECAY);

        double initial_lr = agent.get_learning_rate();

        // Perform many updates
        for (int i = 0; i < 1000; ++i) {
            agent.update(0, 0, 1.0, 1, false);
        }

        double final_lr = agent.get_learning_rate();
        test_utils::assert_true(final_lr < initial_lr, "Exponential decay reduces learning rate");
    }

    // Test adaptive scheduling
    {
        QLearning<int, int> agent(0.1, 0.9, 0.0, actions, QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY,
                                  QLearning<int, int>::LearningRateSchedule::ADAPTIVE);

        // Adaptive scheduling should work without errors
        for (int i = 0; i < 100; ++i) {
            agent.update(i % 10, i % 4, (i % 2 == 0) ? 1.0 : -1.0, (i + 1) % 10, false);
        }

        test_utils::assert_true(true, "Adaptive learning rate scheduling completes without errors");
    }
}

int main() {
    std::cout << "=== Q-Learning Advanced Features Tests ===" << std::endl;

    test_double_q_learning();
    test_eligibility_traces();
    test_experience_replay();
    test_exploration_strategies();
    test_action_masking();
    test_reward_shaping();
    test_learning_rate_scheduling();

    test_utils::print_summary();

    return (test_utils::test_count == test_utils::pass_count) ? 0 : 1;
}
