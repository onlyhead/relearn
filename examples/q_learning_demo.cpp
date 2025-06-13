/**
 * @file advanced_q_learning_demo.cpp
 * @brief Comprehensive demonstration of production-ready Q-learning features
 *
 * This demo showcases:
 * - Multiple exploration strategies
 * - Experience replay
 * - Double Q-learning
 * - Eligibility traces
 * - Performance monitoring
 * - Action masking
 * - Q-table persistence
 * - Multithreaded training
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <relearn/relearn.hpp>
#include <thread>
#include <vector>

using namespace relearn::model_free_value_based;

// Grid world environment configuration
struct GridWorld {
    static constexpr int WIDTH = 10;
    static constexpr int HEIGHT = 10;
    static constexpr int GOAL_STATE = WIDTH * HEIGHT - 1; // Bottom-right corner
    static constexpr int START_STATE = 0;                 // Top-left corner

    // Actions: 0=up, 1=down, 2=left, 3=right
    enum Action { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

    // Get next state given current state and action
    static int get_next_state(int state, int action) {
        int row = state / WIDTH;
        int col = state % WIDTH;

        switch (action) {
        case UP:
            return (row > 0) ? (row - 1) * WIDTH + col : state;
        case DOWN:
            return (row < HEIGHT - 1) ? (row + 1) * WIDTH + col : state;
        case LEFT:
            return (col > 0) ? row * WIDTH + (col - 1) : state;
        case RIGHT:
            return (col < WIDTH - 1) ? row * WIDTH + (col + 1) : state;
        default:
            return state;
        }
    }

    // Get reward for reaching a state
    static double get_reward(int state, int action, int next_state) {
        if (next_state == GOAL_STATE)
            return 100.0;
        if (state == next_state)
            return -1.0; // Wall penalty
        return -0.1;     // Small step cost
    }

    // Check if state is terminal
    static bool is_terminal(int state) { return state == GOAL_STATE; }

    // Action masking - prevent going into walls
    static bool is_action_valid(int state, int action) {
        return get_next_state(state, action) != state || state == GOAL_STATE;
    }
};

void print_section_header(const std::string &title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_statistics(const std::string &name, const auto &stats) {
    std::cout << "\n" << name << " Statistics:\n";
    std::cout << "  Total Updates: " << stats.total_updates << ", Total Actions: " << stats.total_actions << "\n";
    std::cout << "  Cumulative Reward: " << std::fixed << std::setprecision(2) << stats.cumulative_reward << "\n";
    std::cout << "  Average Q-value: " << std::fixed << std::setprecision(4) << stats.average_q_value << "\n";
    std::cout << "  Exploration Ratio: " << std::fixed << std::setprecision(2) << stats.exploration_ratio * 100
              << "%\n";
    std::cout << "  Training Time: " << stats.total_training_time.count() << " ms\n";
}

// Run training episode for Q-learning (unified implementation)
double run_episode(QLearning<int, int> &agent, int max_steps = 1000) {
    int state = GridWorld::START_STATE;
    double total_reward = 0.0;
    int steps = 0;

    while (!GridWorld::is_terminal(state) && steps < max_steps) {
        int action = agent.select_action(state);
        int next_state = GridWorld::get_next_state(state, action);
        double reward = GridWorld::get_reward(state, action, next_state);
        bool terminal = GridWorld::is_terminal(next_state);

        agent.update(state, action, reward, next_state, terminal);

        total_reward += reward;
        state = next_state;
        steps++;
    }

    return total_reward;
}

// Evaluate agent performance
template <typename Agent> double evaluate_agent(Agent &agent, int num_episodes = 100) {
    // Save original epsilon and set to 0 for evaluation
    double original_epsilon = agent.get_epsilon();
    agent.set_epsilon(0.0);

    double total_reward = 0.0;
    for (int i = 0; i < num_episodes; ++i) {
        total_reward += run_episode(agent);
    }

    // Restore original epsilon
    agent.set_epsilon(original_epsilon);

    return total_reward / num_episodes;
}

void demo_basic_q_learning() {
    print_section_header("1. Basic Q-Learning Demo");

    std::vector<int> actions = {0, 1, 2, 3};
    // Create agent with basic configuration (no advanced features enabled)
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    std::cout << "Training basic Q-learning agent...\n";

    // Training
    for (int episode = 0; episode < 1000; ++episode) {
        run_episode(agent);

        if ((episode + 1) % 200 == 0) {
            double avg_reward = evaluate_agent(agent, 10);
            std::cout << "Episode " << (episode + 1) << ", Average Reward: " << std::fixed << std::setprecision(2)
                      << avg_reward << "\n";
        }
    }

    std::cout << "\nFinal Q-table size: " << agent.get_q_table_size() << " states\n";
    std::cout << "Sample Q-values:\n";
    std::cout << "  Q(0,1) = " << std::fixed << std::setprecision(4) << agent.get_q_value(0, 1) << "\n";
    std::cout << "  Q(0,3) = " << std::fixed << std::setprecision(4) << agent.get_q_value(0, 3) << "\n";
}

void demo_advanced_q_learning() {
    print_section_header("2. Advanced Q-Learning Features Demo");

    std::vector<int> actions = {0, 1, 2, 3};

    // Create advanced agent with multiple features
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions, QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY,
                              QLearning<int, int>::LearningRateSchedule::EXPONENTIAL_DECAY);

    // Enable advanced features
    agent.set_double_q_learning(true);
    agent.set_eligibility_traces(true, 0.9);
    agent.set_experience_replay(true, 5000, 64);

    // Set action masking
    agent.set_action_mask([](int state, int action) { return GridWorld::is_action_valid(state, action); });

    // Set reward shaping
    agent.set_reward_shaping([](double reward) {
        return reward * 1.1; // Slight reward amplification
    });

    std::cout << "Training advanced Q-learning agent with:\n";
    std::cout << "  - Double Q-learning\n";
    std::cout << "  - Eligibility traces (λ=0.9)\n";
    std::cout << "  - Experience replay (capacity=5000, batch=64)\n";
    std::cout << "  - Action masking\n";
    std::cout << "  - Reward shaping\n";
    std::cout << "  - Exponential learning rate decay\n\n";

    // Training
    for (int episode = 0; episode < 1000; ++episode) {
        run_episode(agent);

        if ((episode + 1) % 200 == 0) {
            double avg_reward = evaluate_agent(agent, 10);
            std::cout << "Episode " << (episode + 1) << ", Average Reward: " << std::fixed << std::setprecision(2)
                      << avg_reward << "\n";
        }
    }

    // Print final statistics
    print_statistics("Advanced Q-Learning", agent.get_statistics());

    // Demonstrate Q-table persistence
    std::cout << "\nSaving Q-table to 'advanced_q_table.bin'...\n";
    try {
        agent.save_q_table("advanced_q_table.bin");
        std::cout << "Q-table saved successfully!\n";

        // Create new agent and load Q-table
        QLearning<int, int> loaded_agent(0.1, 0.9, 0.0, actions);
        loaded_agent.load_q_table("advanced_q_table.bin");

        double loaded_performance = evaluate_agent(loaded_agent, 10);
        std::cout << "Loaded agent performance: " << std::fixed << std::setprecision(2) << loaded_performance << "\n";

    } catch (const std::exception &e) {
        std::cout << "Note: Q-table persistence requires POD state/action types. Error: " << e.what() << "\n";
    }
}

void demo_exploration_strategies() {
    print_section_header("3. Exploration Strategies Comparison");

    std::vector<int> actions = {0, 1, 2, 3};

    // Test different exploration strategies
    std::vector<std::pair<std::string, QLearning<int, int>::ExplorationStrategy>> strategies = {
        {"Epsilon-Greedy", QLearning<int, int>::ExplorationStrategy::EPSILON_GREEDY},
        {"Boltzmann", QLearning<int, int>::ExplorationStrategy::BOLTZMANN},
        {"UCB1", QLearning<int, int>::ExplorationStrategy::UCB1}};

    for (const auto &[name, strategy] : strategies) {
        std::cout << "\nTesting " << name << " exploration:\n";

        QLearning<int, int> agent(0.1, 0.9, 0.1, actions, strategy);

        // Configure strategy-specific parameters
        if (strategy == QLearning<int, int>::ExplorationStrategy::BOLTZMANN) {
            agent.set_temperature(2.0);
        } else if (strategy == QLearning<int, int>::ExplorationStrategy::UCB1) {
            agent.set_ucb_c(1.4);
        }

        // Train for fewer episodes to see exploration behavior
        for (int episode = 0; episode < 500; ++episode) {
            run_episode(agent);
        }

        double performance = evaluate_agent(agent, 20);
        std::cout << "  Performance: " << std::fixed << std::setprecision(2) << performance;

        auto stats = agent.get_statistics();
        std::cout << ", Exploration Ratio: " << std::fixed << std::setprecision(1) << stats.exploration_ratio * 100
                  << "%\n";
    }
}

void demo_performance_comparison() {
    print_section_header("4. Performance Comparison: Basic vs Advanced");

    std::vector<int> actions = {0, 1, 2, 3};
    const int num_episodes = 800;
    const int eval_interval = 100;

    // Basic Q-learning
    QLearning<int, int> basic_agent(0.1, 0.9, 0.1, actions);

    // Advanced Q-learning with all features
    QLearning<int, int> advanced_agent(0.1, 0.9, 0.1, actions);
    advanced_agent.set_double_q_learning(true);
    advanced_agent.set_eligibility_traces(true, 0.95);
    advanced_agent.set_experience_replay(true, 2000, 32);

    std::vector<double> basic_performance;
    std::vector<double> advanced_performance;

    std::cout << std::setw(10) << "Episode" << std::setw(15) << "Basic Q" << std::setw(15) << "Advanced Q"
              << std::setw(15) << "Improvement\n";
    std::cout << std::string(55, '-') << "\n";

    for (int episode = 0; episode < num_episodes; ++episode) {
        run_episode(basic_agent);
        run_episode(advanced_agent);

        if ((episode + 1) % eval_interval == 0) {
            double basic_perf = evaluate_agent(basic_agent, 20);
            double advanced_perf = evaluate_agent(advanced_agent, 20);

            basic_performance.push_back(basic_perf);
            advanced_performance.push_back(advanced_perf);

            double improvement = ((advanced_perf - basic_perf) / std::abs(basic_perf)) * 100;

            std::cout << std::setw(10) << (episode + 1) << std::setw(15) << std::fixed << std::setprecision(2)
                      << basic_perf << std::setw(15) << std::fixed << std::setprecision(2) << advanced_perf
                      << std::setw(14) << std::fixed << std::setprecision(1) << improvement << "%\n";
        }
    }

    // Final comparison
    double final_basic = basic_performance.back();
    double final_advanced = advanced_performance.back();
    double final_improvement = ((final_advanced - final_basic) / std::abs(final_basic)) * 100;

    std::cout << "\nFinal Performance Summary:\n";
    std::cout << "  Basic Q-Learning: " << std::fixed << std::setprecision(2) << final_basic << "\n";
    std::cout << "  Advanced Q-Learning: " << std::fixed << std::setprecision(2) << final_advanced << "\n";
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1) << final_improvement << "%\n";
}

void demo_multithreaded_training() {
    print_section_header("5. Multithreaded Training Demo");

    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> shared_agent(0.1, 0.9, 0.1, actions);

    const int num_threads = 4;
    const int episodes_per_thread = 250;

    std::cout << "Training with " << num_threads << " threads, " << episodes_per_thread << " episodes each...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&shared_agent, i]() {
            constexpr int episodes = 250;
            for (int episode = 0; episode < episodes; ++episode) {
                run_episode(shared_agent);

                if ((episode + 1) % 50 == 0) {
                    std::cout << "Thread " << i << " completed " << (episode + 1) << " episodes\n";
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double final_performance = evaluate_agent(shared_agent, 50);

    std::cout << "\nMultithreaded Training Results:\n";
    std::cout << "  Total Episodes: " << num_threads * episodes_per_thread << "\n";
    std::cout << "  Training Time: " << duration.count() << " ms\n";
    std::cout << "  Final Performance: " << std::fixed << std::setprecision(2) << final_performance << "\n";

    print_statistics("Multithreaded Agent", shared_agent.get_statistics());
}

int main() {
    std::cout << "Advanced Q-Learning Production Demo\n";
    std::cout << "==================================\n";
    std::cout << "Environment: " << GridWorld::WIDTH << "x" << GridWorld::HEIGHT << " Grid World\n";
    std::cout << "Goal: Navigate from top-left (0) to bottom-right (" << GridWorld::GOAL_STATE << ")\n";

    try {
        demo_basic_q_learning();
        demo_advanced_q_learning();
        demo_exploration_strategies();
        demo_performance_comparison();
        demo_multithreaded_training();

        print_section_header("Demo Complete");
        std::cout << "All advanced Q-learning features demonstrated successfully!\n";
        std::cout << "The library provides production-ready RL capabilities with:\n";
        std::cout << "  ✓ Multiple exploration strategies\n";
        std::cout << "  ✓ Experience replay and eligibility traces\n";
        std::cout << "  ✓ Double Q-learning for reduced overestimation\n";
        std::cout << "  ✓ Learning rate scheduling and action masking\n";
        std::cout << "  ✓ Performance monitoring and Q-table persistence\n";
        std::cout << "  ✓ Thread-safe operations for parallel training\n";

    } catch (const std::exception &e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
