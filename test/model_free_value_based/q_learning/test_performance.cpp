#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <relearn/relearn.hpp>
#include <thread>
#include <vector>

using namespace relearn::model_free_value_based;

// Performance testing framework
namespace perf_utils {
    void time_operation(const std::string &operation_name, std::function<void()> operation) {
        auto start = std::chrono::high_resolution_clock::now();
        operation();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << operation_name << ": " << duration.count() << " microseconds" << std::endl;
    }

    double measure_throughput(const std::string &operation_name, std::function<void()> operation, int iterations) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            operation();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double ops_per_second = (double)iterations / (duration.count() / 1e6);
        std::cout << operation_name << ": " << ops_per_second << " ops/sec (" << iterations << " iterations in "
                  << duration.count() << " μs)" << std::endl;

        return ops_per_second;
    }
} // namespace perf_utils

void test_update_performance() {
    std::cout << "\n--- Testing Update Performance ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    // Test basic update performance
    int iterations = 100000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> state_dist(0, 999);
    std::uniform_int_distribution<> action_dist(0, 3);
    std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

    double update_throughput = perf_utils::measure_throughput(
        "Basic Q-Learning Updates",
        [&]() {
            int state = state_dist(gen);
            int action = action_dist(gen);
            double reward = reward_dist(gen);
            int next_state = state_dist(gen);
            agent.update(state, action, reward, next_state, false);
        },
        iterations);

    // Should achieve >100k updates/sec as mentioned in README
    if (update_throughput > 100000) {
        std::cout << "✓ PASS: Achieves target performance (>100k updates/sec)" << std::endl;
    } else {
        std::cout << "⚠ WARNING: Below target performance (<100k updates/sec)" << std::endl;
    }
}

void test_action_selection_performance() {
    std::cout << "\n--- Testing Action Selection Performance ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    // Pre-populate with some Q-values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> state_dist(0, 999);
    std::uniform_int_distribution<> action_dist(0, 3);
    std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

    for (int i = 0; i < 10000; ++i) {
        agent.update(state_dist(gen), action_dist(gen), reward_dist(gen), state_dist(gen), false);
    }

    // Test action selection performance
    int iterations = 100000;
    double selection_throughput = perf_utils::measure_throughput(
        "Action Selection",
        [&]() {
            int state = state_dist(gen);
            agent.select_action(state);
        },
        iterations);

    std::cout << "Action selection throughput: " << selection_throughput << " selections/sec" << std::endl;
}

void test_advanced_features_performance() {
    std::cout << "\n--- Testing Advanced Features Performance ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};

    // Test double Q-learning performance
    {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
        agent.set_double_q_learning(true);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> state_dist(0, 99);
        std::uniform_int_distribution<> action_dist(0, 3);
        std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

        double double_q_throughput = perf_utils::measure_throughput(
            "Double Q-Learning Updates",
            [&]() { agent.update(state_dist(gen), action_dist(gen), reward_dist(gen), state_dist(gen), false); },
            10000);
    }

    // Test eligibility traces performance
    {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
        agent.set_eligibility_traces(true, 0.9);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> state_dist(0, 99);
        std::uniform_int_distribution<> action_dist(0, 3);
        std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

        double traces_throughput = perf_utils::measure_throughput(
            "Eligibility Traces Updates",
            [&]() { agent.update(state_dist(gen), action_dist(gen), reward_dist(gen), state_dist(gen), false); },
            10000);
    }

    // Test experience replay performance
    {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
        agent.set_experience_replay(true, 5000, 32);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> state_dist(0, 99);
        std::uniform_int_distribution<> action_dist(0, 3);
        std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

        double replay_throughput = perf_utils::measure_throughput(
            "Experience Replay Updates",
            [&]() { agent.update(state_dist(gen), action_dist(gen), reward_dist(gen), state_dist(gen), false); },
            10000);
    }
}

void test_memory_efficiency() {
    std::cout << "\n--- Testing Memory Efficiency ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    // Test Q-table size tracking
    size_t initial_size = agent.get_q_table_size();
    std::cout << "Initial Q-table size: " << initial_size << std::endl;

    // Add many state-action pairs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> state_dist(0, 9999);
    std::uniform_int_distribution<> action_dist(0, 3);
    std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

    for (int i = 0; i < 10000; ++i) {
        agent.update(state_dist(gen), action_dist(gen), reward_dist(gen), state_dist(gen), false);
    }

    size_t final_size = agent.get_q_table_size();
    std::cout << "Final Q-table size: " << final_size << std::endl;
    std::cout << "Memory growth: " << (final_size - initial_size) << " entries" << std::endl;

    // Verify sparse representation (should not have 10000 * 4 = 40000 entries)
    if (final_size < 40000) {
        std::cout << "✓ PASS: Sparse Q-table representation working efficiently" << std::endl;
    } else {
        std::cout << "⚠ WARNING: Q-table may not be using sparse representation efficiently" << std::endl;
    }
}

void test_thread_safety_performance() {
    std::cout << "\n--- Testing Thread Safety Performance ---" << std::endl;
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    const int num_threads = 4;
    const int updates_per_thread = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&agent, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t); // Different seed per thread
            std::uniform_int_distribution<> state_dist(0, 999);
            std::uniform_int_distribution<> action_dist(0, 3);
            std::uniform_real_distribution<> reward_dist(-1.0, 1.0);

            for (int i = 0; i < 10000; ++i) {
                agent.update(state_dist(gen), action_dist(gen), reward_dist(gen), state_dist(gen), false);

                // Mix in some action selections
                if (i % 10 == 0) {
                    agent.select_action(state_dist(gen));
                }
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    int total_operations = num_threads * updates_per_thread;
    double ops_per_second = (double)total_operations / (duration.count() / 1e6);

    std::cout << "Multi-threaded performance: " << ops_per_second << " ops/sec" << std::endl;
    std::cout << "Using " << num_threads << " threads, " << updates_per_thread << " updates each" << std::endl;

    // Verify no corruption occurred
    auto stats = agent.get_statistics();
    std::cout << "Final statistics - Total updates: " << stats.total_updates << std::endl;

    if (stats.total_updates == total_operations) {
        std::cout << "✓ PASS: Thread safety maintained - all updates recorded" << std::endl;
    } else {
        std::cout << "⚠ WARNING: Thread safety issue - update count mismatch" << std::endl;
    }
}

int main() {
    std::cout << "=== Q-Learning Performance Tests ===" << std::endl;

    test_update_performance();
    test_action_selection_performance();
    test_advanced_features_performance();
    test_memory_efficiency();
    test_thread_safety_performance();

    std::cout << "\n=== Performance Test Complete ===" << std::endl;
    std::cout << "Note: Performance may vary based on hardware and system load." << std::endl;

    return 0;
}
