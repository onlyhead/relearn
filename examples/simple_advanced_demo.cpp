/**
 * @file simple_advanced_demo.cpp
 * @brief Simplified demonstration to debug the hanging issue
 */

#include <chrono>
#include <iostream>
#include <relearn/relearn.hpp>
#include <vector>

using namespace relearn::model_free_value_based;

// Simple environment for debugging
struct SimpleEnv {
    static constexpr int GOAL_STATE = 5;
    static constexpr int START_STATE = 0;

    static int get_next_state(int state, int action) {
        // Simple linear environment: 0 -> 1 -> 2 -> 3 -> 4 -> 5
        if (action == 0 && state > 0)
            return state - 1; // go left
        if (action == 1 && state < 5)
            return state + 1; // go right
        return state;         // stay in place if invalid
    }

    static double get_reward(int state, int action, int next_state) {
        if (next_state == GOAL_STATE)
            return 10.0;
        return -0.1;
    }

    static bool is_terminal(int state) { return state == GOAL_STATE; }
};

double run_simple_episode(QLearning<int, int> &agent, int max_steps = 100) {
    int state = SimpleEnv::START_STATE;
    double total_reward = 0.0;
    int steps = 0;

    while (!SimpleEnv::is_terminal(state) && steps < max_steps) {
        int action = agent.select_action(state);
        int next_state = SimpleEnv::get_next_state(state, action);
        double reward = SimpleEnv::get_reward(state, action, next_state);
        bool terminal = SimpleEnv::is_terminal(next_state);

        std::cout << "Step " << steps << ": State " << state << " -> Action " << action << " -> Next State "
                  << next_state << " (reward: " << reward << ")\n";

        agent.update(state, action, reward, next_state, terminal);

        total_reward += reward;
        state = next_state;
        steps++;

        // Add a small delay to see what's happening
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return total_reward;
}

int main() {
    std::cout << "Simple Advanced Q-Learning Debug Demo\n";
    std::cout << "=====================================\n";

    std::vector<int> actions = {0, 1}; // left, right

    try {
        // Test 1: Basic advanced Q-learning without extra features
        std::cout << "\nTest 1: Basic Advanced Q-Learning (no extra features)\n";
        QLearning<int, int> basic_agent(0.1, 0.9, 0.3, actions);

        for (int episode = 0; episode < 3; ++episode) {
            std::cout << "\n--- Episode " << (episode + 1) << " ---\n";
            double reward = run_simple_episode(basic_agent, 20);
            std::cout << "Episode reward: " << reward << "\n";
        }

        // Test 2: With double Q-learning only
        std::cout << "\nTest 2: With Double Q-Learning\n";
        QLearning<int, int> double_agent(0.1, 0.9, 0.3, actions);
        double_agent.set_double_q_learning(true);

        for (int episode = 0; episode < 2; ++episode) {
            std::cout << "\n--- Episode " << (episode + 1) << " ---\n";
            double reward = run_simple_episode(double_agent, 20);
            std::cout << "Episode reward: " << reward << "\n";
        }

        // Test 3: With eligibility traces only
        std::cout << "\nTest 3: With Eligibility Traces\n";
        QLearning<int, int> trace_agent(0.1, 0.9, 0.3, actions);
        trace_agent.set_eligibility_traces(true, 0.9);

        for (int episode = 0; episode < 2; ++episode) {
            std::cout << "\n--- Episode " << (episode + 1) << " ---\n";
            double reward = run_simple_episode(trace_agent, 20);
            std::cout << "Episode reward: " << reward << "\n";
        }

        // Test 4: With experience replay only
        std::cout << "\nTest 4: With Experience Replay\n";
        QLearning<int, int> replay_agent(0.1, 0.9, 0.3, actions);
        replay_agent.set_experience_replay(true, 100, 8);

        for (int episode = 0; episode < 2; ++episode) {
            std::cout << "\n--- Episode " << (episode + 1) << " ---\n";
            double reward = run_simple_episode(replay_agent, 20);
            std::cout << "Episode reward: " << reward << "\n";
        }

        std::cout << "\nAll tests completed successfully!\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
