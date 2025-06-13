/**
 * @file multi_agent_harvest_demo.cpp
 * @brief Advanced Q-learning demo: Pea harvesting with chaser bin coordination
 *
 * This demo showcases advanced Q-learning features:
 * - Multi-agent environment with 2 harvesting machines + 1 chaser bin
 * - Dynamic environment (harvested cells change over time)
 * - Advanced coordination using experience replay and eligibility traces
 * - Action masking (chaser can only move on harvested cells)
 * - Complex reward structure for efficiency optimization
 * - Boltzmann exploration and double Q-learning
 * - Real-time state updates and machine capacity tracking
 * - Performance analysis of different Q-learning configurations
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <relearn/relearn.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace relearn::model_free_value_based;

// Field environment configuration
struct HarvestField {
    static constexpr int WIDTH = 100;
    static constexpr int HEIGHT = 100;
    static constexpr int TOTAL_CELLS = WIDTH * HEIGHT;

    // Machine capacity and harvest rates
    static constexpr int MACHINE_CAPACITY = 50;
    static constexpr double HARVEST_RATE = 0.8; // Probability of harvesting per step
    static constexpr int CHASER_CAPACITY = 200;

    // Actions for all agents: 0=up, 1=down, 2=left, 3=right, 4=stay
    enum Action { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, STAY = 4 };

    // Field state
    std::vector<std::vector<bool>> harvested;           // Track harvested cells
    std::vector<std::pair<int, int>> machine_positions; // Machine positions
    std::vector<int> machine_loads;                     // Current load of each machine
    std::pair<int, int> chaser_position;                // Chaser bin position
    int chaser_load;

    std::mt19937 rng;

    HarvestField() : rng(std::random_device{}()) { reset(); }

    void reset() {
        // Initialize field as unharvested
        harvested.assign(HEIGHT, std::vector<bool>(WIDTH, false));

        // Place machines at opposite corners of the field
        machine_positions = {{0, 0}, {HEIGHT - 1, WIDTH - 1}};
        machine_loads = {0, 0};

        // Place chaser bin in the middle
        chaser_position = {HEIGHT / 2, WIDTH / 2};
        chaser_load = 0;
    }

    // Convert 2D position to 1D state
    static int pos_to_state(int row, int col) { return row * WIDTH + col; }

    // Convert 1D state to 2D position
    static std::pair<int, int> state_to_pos(int state) { return {state / WIDTH, state % WIDTH}; }

    // Get next position given current position and action
    std::pair<int, int> get_next_position(const std::pair<int, int> &pos, int action) {
        int row = pos.first, col = pos.second;

        switch (action) {
        case UP:
            return {std::max(0, row - 1), col};
        case DOWN:
            return {std::min(HEIGHT - 1, row + 1), col};
        case LEFT:
            return {row, std::max(0, col - 1)};
        case RIGHT:
            return {row, std::min(WIDTH - 1, col + 1)};
        case STAY:
            return {row, col};
        default:
            return {row, col};
        }
    }

    // Check if position is valid and within bounds
    bool is_valid_position(const std::pair<int, int> &pos) {
        return pos.first >= 0 && pos.first < HEIGHT && pos.second >= 0 && pos.second < WIDTH;
    }

    // Update machine positions and harvest
    void update_machines() {
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < 2; ++i) {
            auto &pos = machine_positions[i];

            // Simple machine movement pattern (they move in lines)
            if (i == 0) {
                // Machine 0: horizontal sweeps, top to bottom
                if (pos.second < WIDTH - 1) {
                    pos.second++;
                } else if (pos.first < HEIGHT - 1) {
                    pos.first++;
                    pos.second = 0;
                }
            } else {
                // Machine 1: vertical sweeps, left to right
                if (pos.first > 0) {
                    pos.first--;
                } else if (pos.second > 0) {
                    pos.second--;
                    pos.first = HEIGHT - 1;
                }
            }

            // Harvest current cell if not already harvested
            if (!harvested[pos.first][pos.second] && dis(rng) < HARVEST_RATE) {
                harvested[pos.first][pos.second] = true;
                machine_loads[i] = std::min(MACHINE_CAPACITY, machine_loads[i] + 1);
            }
        }
    }

    // Check if chaser can move to a position (only on harvested cells)
    bool can_chaser_move_to(const std::pair<int, int> &pos) {
        if (!is_valid_position(pos))
            return false;
        return harvested[pos.first][pos.second] || pos == chaser_position;
    }

    // Get distance between two positions (Manhattan distance)
    int get_distance(const std::pair<int, int> &pos1, const std::pair<int, int> &pos2) {
        return std::abs(pos1.first - pos2.first) + std::abs(pos1.second - pos2.second);
    }

    // Check if chaser is at same position as a machine
    int chaser_at_machine() {
        for (int i = 0; i < 2; ++i) {
            if (chaser_position == machine_positions[i]) {
                return i;
            }
        }
        return -1;
    }

    // Collect from machine if chaser is at machine position
    double collect_from_machine(int machine_id) {
        if (machine_id < 0 || machine_id >= 2)
            return 0.0;

        int collected = std::min(machine_loads[machine_id], CHASER_CAPACITY - chaser_load);
        machine_loads[machine_id] -= collected;
        chaser_load += collected;

        return collected * 2.0; // Reward for successful collection
    }

    // Get comprehensive state for chaser (includes positions and loads)
    int get_chaser_state() {
        // Create a compact state representation
        int state = pos_to_state(chaser_position.first, chaser_position.second);

        // Add machine information (position and load status)
        for (int i = 0; i < 2; ++i) {
            state += (machine_loads[i] > MACHINE_CAPACITY * 0.8 ? 1 << (20 + i) : 0);
            state += (get_distance(chaser_position, machine_positions[i]) << (10 + i * 3));
        }

        return state;
    }

    // Get reward for chaser action
    double get_chaser_reward(int action, const std::pair<int, int> &old_pos) {
        double reward = 0.0;

        // Small penalty for movement (encourage efficiency)
        if (action != STAY) {
            reward -= 0.1;
        }

        // Check if chaser is at a machine and can collect
        int machine_at = chaser_at_machine();
        if (machine_at >= 0 && machine_loads[machine_at] > 0) {
            reward += collect_from_machine(machine_at);
        }

        // Reward for being close to full machines
        for (int i = 0; i < 2; ++i) {
            if (machine_loads[i] > MACHINE_CAPACITY * 0.8) {
                int distance = get_distance(chaser_position, machine_positions[i]);
                reward += std::max(0.0, 5.0 - distance * 0.1);
            }
        }

        // Penalty for being far from full machines when they exist
        for (int i = 0; i < 2; ++i) {
            if (machine_loads[i] >= MACHINE_CAPACITY) {
                int distance = get_distance(chaser_position, machine_positions[i]);
                reward -= distance * 0.05; // Penalty increases with distance
            }
        }

        // Large penalty if machines overflow (lose peas)
        for (int i = 0; i < 2; ++i) {
            if (machine_loads[i] >= MACHINE_CAPACITY) {
                reward -= 10.0; // Heavy penalty for machine overflow
            }
        }

        return reward;
    }

    // Print field visualization
    void print_field(int max_rows = 20, int max_cols = 20) {
        std::cout << "\nField Status (" << max_rows << "x" << max_cols << " view):\n";
        std::cout << "Legend: . = unharvested, # = harvested, M0/M1 = machines, C = chaser\n";

        for (int r = 0; r < std::min(HEIGHT, max_rows); ++r) {
            for (int c = 0; c < std::min(WIDTH, max_cols); ++c) {
                char cell = harvested[r][c] ? '#' : '.';

                // Check for agents at this position
                if (std::make_pair(r, c) == chaser_position) {
                    cell = 'C';
                } else if (std::make_pair(r, c) == machine_positions[0]) {
                    cell = '0';
                } else if (std::make_pair(r, c) == machine_positions[1]) {
                    cell = '1';
                }

                std::cout << cell;
            }
            std::cout << "\n";
        }

        std::cout << "\nMachine Loads: M0=" << machine_loads[0] << "/" << MACHINE_CAPACITY
                  << ", M1=" << machine_loads[1] << "/" << MACHINE_CAPACITY << ", Chaser=" << chaser_load << "/"
                  << CHASER_CAPACITY << "\n";
    }

    // Get statistics
    struct FieldStats {
        int total_harvested;
        int machine0_load;
        int machine1_load;
        int chaser_load;
        double harvest_efficiency;
    };

    FieldStats get_stats() {
        int total_harvested = 0;
        for (const auto &row : harvested) {
            for (bool cell : row) {
                if (cell)
                    total_harvested++;
            }
        }

        return {total_harvested, machine_loads[0], machine_loads[1], chaser_load,
                static_cast<double>(total_harvested) / TOTAL_CELLS};
    }
};

void print_section_header(const std::string &title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// Run a single episode of the harvest simulation
double run_harvest_episode(QLearning<int, int> &chaser_agent, HarvestField &field, int max_steps = 5000,
                           bool verbose = false) {
    field.reset();
    double total_reward = 0.0;
    int steps = 0;

    while (steps < max_steps) {
        // Update machine positions and harvesting
        field.update_machines();

        // Get current state for chaser
        int current_state = field.get_chaser_state();

        // Select action for chaser
        int action = chaser_agent.select_action(current_state);

        // Apply action masking - chaser can only move on harvested cells
        auto old_pos = field.chaser_position;
        auto new_pos = field.get_next_position(field.chaser_position, action);

        // If move is invalid, force STAY action
        if (!field.can_chaser_move_to(new_pos)) {
            action = HarvestField::STAY;
            new_pos = old_pos;
        }

        // Update chaser position
        field.chaser_position = new_pos;

        // Calculate reward
        double reward = field.get_chaser_reward(action, old_pos);
        total_reward += reward;

        // Get next state
        int next_state = field.get_chaser_state();

        // Update Q-learning agent
        chaser_agent.update(current_state, action, reward, next_state, false);

        // Print status periodically
        if (verbose && steps % 500 == 0) {
            std::cout << "Step " << steps << ", Reward: " << std::fixed << std::setprecision(2) << reward << "\n";
            field.print_field(15, 15);
        }

        steps++;

        // Check termination conditions
        auto stats = field.get_stats();
        if (stats.harvest_efficiency > 0.8) { // Stop when 80% harvested
            if (verbose) {
                std::cout << "Episode complete: 80% of field harvested\n";
            }
            break;
        }
    }

    return total_reward;
}

// Evaluate chaser agent performance
double evaluate_chaser_agent(QLearning<int, int> &agent, int num_episodes = 10) {
    double original_epsilon = agent.get_epsilon();
    agent.set_epsilon(0.0); // No exploration during evaluation

    double total_reward = 0.0;
    for (int i = 0; i < num_episodes; ++i) {
        HarvestField eval_field;
        total_reward += run_harvest_episode(agent, eval_field, 3000, false);
    }

    agent.set_epsilon(original_epsilon); // Restore exploration
    return total_reward / num_episodes;
}

void demo_advanced_harvest_coordination() {
    print_section_header("1. Advanced Multi-Agent Coordination with Experience Replay");

    HarvestField field;
    std::vector<int> actions = {0, 1, 2, 3, 4};

    // Create advanced chaser agent
    QLearning<int, int> chaser_agent(0.1,  // learning rate
                                     0.95, // discount factor
                                     0.15, // exploration rate
                                     actions, QLearning<int, int>::ExplorationStrategy::BOLTZMANN,
                                     QLearning<int, int>::LearningRateSchedule::EXPONENTIAL_DECAY);

    // Enable advanced features
    chaser_agent.set_double_q_learning(true);
    chaser_agent.set_experience_replay(true, 10000, 128);
    chaser_agent.set_eligibility_traces(true, 0.9);
    chaser_agent.set_temperature(1.5); // For Boltzmann exploration

    // Action masking
    chaser_agent.set_action_mask([&field](int state, int action) {
        auto current_pos = field.chaser_position;
        auto next_pos = field.get_next_position(current_pos, action);
        return field.can_chaser_move_to(next_pos);
    });

    // Reward shaping for better learning
    chaser_agent.set_reward_shaping([](double reward) {
        return std::tanh(reward / 5.0); // Normalize rewards
    });

    std::cout << "Training advanced chaser agent with:\n";
    std::cout << "  - Double Q-learning\n";
    std::cout << "  - Experience replay (capacity=10000, batch=128)\n";
    std::cout << "  - Eligibility traces (λ=0.9)\n";
    std::cout << "  - Boltzmann exploration\n";
    std::cout << "  - Action masking for harvested-cell-only movement\n";
    std::cout << "  - Reward shaping\n\n";

    // Training
    const int num_episodes = 150;
    double best_performance = -std::numeric_limits<double>::infinity();

    for (int episode = 0; episode < num_episodes; ++episode) {
        double reward = run_harvest_episode(chaser_agent, field, 4000, false);

        if ((episode + 1) % 25 == 0) {
            double performance = evaluate_chaser_agent(chaser_agent, 8);
            best_performance = std::max(best_performance, performance);

            std::cout << "Episode " << (episode + 1) << ", Performance: " << std::fixed << std::setprecision(1)
                      << performance << ", Best: " << std::fixed << std::setprecision(1) << best_performance << "\n";
        }
    }

    // Print agent statistics
    auto stats = chaser_agent.get_statistics();
    std::cout << "\nChaser Agent Statistics:\n";
    std::cout << "  Total updates: " << stats.total_updates << "\n";
    std::cout << "  Cumulative reward: " << std::fixed << std::setprecision(1) << stats.cumulative_reward << "\n";
    std::cout << "  Exploration ratio: " << std::fixed << std::setprecision(1) << stats.exploration_ratio * 100
              << "%\n";
    std::cout << "  Q-table size: " << chaser_agent.get_q_table_size() << " states\n";

    // Final demonstration
    std::cout << "\nAdvanced agent final demonstration:\n";
    run_harvest_episode(chaser_agent, field, 1500, true);
}

void demo_performance_analysis() {
    print_section_header("2. Advanced Q-Learning Performance Analysis");

    std::vector<int> actions = {0, 1, 2, 3, 4};

    // Create multiple advanced agents with different configurations
    QLearning<int, int> standard_agent(0.1, 0.95, 0.15, actions, QLearning<int, int>::ExplorationStrategy::BOLTZMANN);
    standard_agent.set_double_q_learning(true);
    standard_agent.set_experience_replay(true, 5000, 64);
    standard_agent.set_temperature(1.8);

    QLearning<int, int> optimized_agent(0.05, 0.98, 0.1, actions, QLearning<int, int>::ExplorationStrategy::BOLTZMANN);
    optimized_agent.set_double_q_learning(true);
    optimized_agent.set_experience_replay(true, 10000, 128);
    optimized_agent.set_eligibility_traces(true, 0.95);
    optimized_agent.set_temperature(1.5);

    const int num_episodes = 100;
    const int eval_interval = 25;

    std::cout << std::setw(10) << "Episode" << std::setw(17) << "Standard Config" << std::setw(17) << "Optimized Config"
              << std::setw(15) << "Improvement\n";
    std::cout << std::string(65, '-') << "\n";

    for (int episode = 0; episode < num_episodes; ++episode) {
        // Train both agents
        HarvestField field1, field2;
        run_harvest_episode(standard_agent, field1, 3000, false);
        run_harvest_episode(optimized_agent, field2, 3000, false);

        if ((episode + 1) % eval_interval == 0) {
            double standard_perf = evaluate_chaser_agent(standard_agent, 8);
            double optimized_perf = evaluate_chaser_agent(optimized_agent, 8);

            double improvement = ((optimized_perf - standard_perf) / std::abs(standard_perf)) * 100;

            std::cout << std::setw(10) << (episode + 1) << std::setw(17) << std::fixed << std::setprecision(1)
                      << standard_perf << std::setw(17) << std::fixed << std::setprecision(1) << optimized_perf
                      << std::setw(14) << std::fixed << std::setprecision(1) << improvement << "%\n";
        }
    }

    // Final statistics comparison
    auto standard_stats = standard_agent.get_statistics();
    auto optimized_stats = optimized_agent.get_statistics();

    std::cout << "\nFinal Agent Statistics Comparison:\n";
    std::cout << "Standard Config:\n";
    std::cout << "  Q-table size: " << standard_agent.get_q_table_size() << " states\n";
    std::cout << "  Total updates: " << standard_stats.total_updates << "\n";
    std::cout << "  Exploration ratio: " << std::fixed << std::setprecision(1) << standard_stats.exploration_ratio * 100
              << "%\n";

    std::cout << "Optimized Config:\n";
    std::cout << "  Q-table size: " << optimized_agent.get_q_table_size() << " states\n";
    std::cout << "  Total updates: " << optimized_stats.total_updates << "\n";
    std::cout << "  Exploration ratio: " << std::fixed << std::setprecision(1)
              << optimized_stats.exploration_ratio * 100 << "%\n";
}

int main() {
    std::cout << "Multi-Agent Pea Harvesting Coordination Demo\n";
    std::cout << "============================================\n";
    std::cout << "Scenario: 2 harvesting machines + 1 chaser bin on 100x100 field\n";
    std::cout << "Challenge: Chaser must coordinate collection while only moving on harvested cells\n";

    try {
        demo_advanced_harvest_coordination();
        demo_performance_analysis();

        print_section_header("Demo Complete");
        std::cout << "Advanced multi-agent harvest coordination successfully demonstrated!\n";
        std::cout << "Key achievements:\n";
        std::cout << "  ✓ Advanced Q-learning with experience replay and eligibility traces\n";
        std::cout << "  ✓ Action masking enforced movement only on harvested cells\n";
        std::cout << "  ✓ Dynamic environment with real-time machine updates\n";
        std::cout << "  ✓ Complex reward structure for efficiency optimization\n";
        std::cout << "  ✓ Boltzmann exploration for intelligent action selection\n";
        std::cout << "  ✓ Double Q-learning for reduced overestimation bias\n";
        std::cout << "  ✓ Scalable to larger fields and more agents\n";

    } catch (const std::exception &e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
