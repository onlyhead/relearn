/**
 * @file multi_agent_harvest_demo.cpp
 * @brief Advanced Q-learning demo: Multi-agent pea harvesting with density-based coordination
 *
 * This demo showcases advanced Q-learning features with realistic constraints:
 * - 3 identical harvesting machines with density-dependent timing
 * - 100x100 field with outer boundary accessible to chaser
 * - Penalty-based movement (chaser penalized for moving on unharvested cells)
 * - Complex coordination with asynchronous machine operation
 * - Territorial division: each machine handles one third of the field
 * - Realistic density-based harvesting times (all machines identical speed)
 * - Advanced Q-learning with experience replay, eligibility traces, and action masking
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

// Complex field environment configuration
struct ComplexHarvestField {
    static constexpr int WIDTH = 500;
    static constexpr int HEIGHT = 500;
    static constexpr int TOTAL_CELLS = WIDTH * HEIGHT;
    static constexpr int BOUNDARY_WIDTH = 2; // Outer boundary always accessible

    // Machine capacity and harvest rates
    static constexpr int MACHINE_CAPACITY = 100;
    static constexpr double BASE_HARVEST_RATE = 0.7; // Base probability of harvesting per step
    static constexpr int CHASER_CAPACITY = 500;
    static constexpr int NUM_MACHINES = 3;

    // Movement penalties and rewards
    static constexpr double INVALID_MOVE_PENALTY = -5.0;
    static constexpr double COLLECTION_REWARD = 3.0;
    static constexpr double MOVEMENT_COST = -0.2;
    static constexpr double OVERFLOW_PENALTY = -20.0;

    // Actions for all agents: 0=up, 1=down, 2=left, 3=right, 4=stay
    enum Action { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, STAY = 4 };

    // Machine configuration
    struct Machine {
        std::pair<int, int> position;
        int load;
        double speed_multiplier;     // Speed variation (0.5 to 1.5)
        double harvest_rate;         // Individual harvest rate
        int territory_start;         // Starting column for this machine's territory
        int territory_end;           // Ending column for this machine's territory
        int direction;               // Current movement direction (1 = right/down, -1 = left/up)
        bool horizontal_sweep;       // Currently doing horizontal sweep
        int steps_since_move;        // For speed control
        int move_interval;           // Steps between moves
        int harvesting_time_left;    // Time left to harvest current cell
        double current_cell_density; // Density of cell being harvested
    };

    // Field state
    std::vector<std::vector<double>> harvest_density; // Harvest density per cell (0.0-1.0)
    std::vector<std::vector<bool>> boundary_cells;    // Track boundary accessibility
    std::vector<Machine> machines;                    // Machine configurations
    std::pair<int, int> chaser_position;              // Chaser bin position
    int chaser_load;

    std::mt19937 rng;
    std::uniform_real_distribution<> speed_dist;
    std::uniform_real_distribution<> harvest_dist;
    std::uniform_real_distribution<> density_dist;

    ComplexHarvestField()
        : rng(std::random_device{}()), speed_dist(0.5, 1.5), harvest_dist(0.0, 1.0), density_dist(0.0, 1.0) {
        reset();
    }

    void reset() {
        // Initialize field with random harvest densities
        harvest_density.assign(HEIGHT, std::vector<double>(WIDTH, 0.0));
        boundary_cells.assign(HEIGHT, std::vector<bool>(WIDTH, false));

        // Generate random harvest densities for inner field
        for (int r = BOUNDARY_WIDTH; r < HEIGHT - BOUNDARY_WIDTH; ++r) {
            for (int c = BOUNDARY_WIDTH; c < WIDTH - BOUNDARY_WIDTH; ++c) {
                harvest_density[r][c] = density_dist(rng);
            }
        }

        // Mark boundary cells as always accessible (outer ring)
        for (int r = 0; r < HEIGHT; ++r) {
            for (int c = 0; c < WIDTH; ++c) {
                if (r < BOUNDARY_WIDTH || r >= HEIGHT - BOUNDARY_WIDTH || c < BOUNDARY_WIDTH ||
                    c >= WIDTH - BOUNDARY_WIDTH) {
                    boundary_cells[r][c] = true;
                }
            }
        }

        // Initialize machines with random speeds and territories
        machines.clear();
        machines.resize(NUM_MACHINES);

        int territory_width = (WIDTH - 2 * BOUNDARY_WIDTH) / NUM_MACHINES;

        for (int i = 0; i < NUM_MACHINES; ++i) {
            Machine &machine = machines[i];

            // Assign territory (each machine gets 1/3 of the inner field)
            machine.territory_start = BOUNDARY_WIDTH + i * territory_width;
            machine.territory_end =
                (i == NUM_MACHINES - 1) ? WIDTH - BOUNDARY_WIDTH : BOUNDARY_WIDTH + (i + 1) * territory_width;

            // Random starting position within territory
            std::uniform_int_distribution<> row_dist(BOUNDARY_WIDTH, HEIGHT - BOUNDARY_WIDTH - 1);
            std::uniform_int_distribution<> col_dist(machine.territory_start, machine.territory_end - 1);

            machine.position = {row_dist(rng), col_dist(rng)};
            machine.load = 0;
            machine.speed_multiplier = 1.0;           // All machines have identical speed
            machine.harvest_rate = BASE_HARVEST_RATE; // All machines have same harvest rate
            machine.direction = (rng() % 2 == 0) ? 1 : -1;
            machine.horizontal_sweep = (rng() % 2 == 0);
            machine.steps_since_move = 0;
            machine.move_interval = 1; // All machines move every step when not harvesting
            machine.harvesting_time_left = 0;
            machine.current_cell_density = 0.0;

            std::cout << "Machine " << i << " initialized:\n";
            std::cout << "  Territory: cols " << machine.territory_start << "-" << machine.territory_end << "\n";
            std::cout << "  Starting position: (" << machine.position.first << ", " << machine.position.second << ")\n";
            std::cout << "  Movement pattern: " << (machine.horizontal_sweep ? "Horizontal" : "Vertical")
                      << " sweep\n\n";
        }

        // Place chaser bin at random boundary position
        std::vector<std::pair<int, int>> boundary_positions;
        for (int r = 0; r < HEIGHT; ++r) {
            for (int c = 0; c < WIDTH; ++c) {
                if (boundary_cells[r][c]) {
                    boundary_positions.push_back({r, c});
                }
            }
        }

        if (!boundary_positions.empty()) {
            std::uniform_int_distribution<> pos_dist(0, boundary_positions.size() - 1);
            chaser_position = boundary_positions[pos_dist(rng)];
        } else {
            chaser_position = {0, 0}; // Fallback
        }

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

    // Check if chaser can move to a position (boundary OR harvested cells)
    bool can_chaser_move_to(const std::pair<int, int> &pos) {
        if (!is_valid_position(pos))
            return false;
        return boundary_cells[pos.first][pos.second] ||
               harvest_density[pos.first][pos.second] < 0.1 || // Cell is mostly harvested
               pos == chaser_position;
    }

    // Update machine positions and harvest with density-based timing only
    void update_machines() {
        for (int i = 0; i < NUM_MACHINES; ++i) {
            Machine &machine = machines[i];
            auto &pos = machine.position;

            // If machine is currently harvesting
            if (machine.harvesting_time_left > 0) {
                machine.harvesting_time_left--;

                // Continue harvesting current cell
                if (harvest_density[pos.first][pos.second] > 0.01) {
                    double harvest_amount = machine.harvest_rate * machine.current_cell_density * 0.1;
                    harvest_density[pos.first][pos.second] =
                        std::max(0.0, harvest_density[pos.first][pos.second] - harvest_amount);

                    // Add to machine load proportional to what was harvested
                    int load_increase = static_cast<int>(harvest_amount * 100);
                    machine.load = std::min(MACHINE_CAPACITY, machine.load + load_increase);
                }

                // If harvesting is complete or cell is depleted, can move next step
                if (machine.harvesting_time_left == 0 || harvest_density[pos.first][pos.second] <= 0.01) {
                    machine.harvesting_time_left = 0;
                }
                continue; // Don't move while harvesting
            }

            // All machines move at the same rate - only density affects timing
            // Check if current cell has harvest density worth working on
            double current_density = harvest_density[pos.first][pos.second];
            if (current_density > 0.1) {
                // Start harvesting current cell
                machine.current_cell_density = current_density;
                // Harvesting time based purely on density (1-20 steps)
                machine.harvesting_time_left = static_cast<int>(1 + current_density * 19);
                continue; // Start harvesting, don't move
            }

            // Move to next position since current cell doesn't need harvesting
            // Complex movement pattern within territory
            if (machine.horizontal_sweep) {
                // Horizontal sweep within territory
                int next_col = pos.second + machine.direction;
                if (next_col >= machine.territory_end || next_col < machine.territory_start) {
                    // Change row and reverse direction
                    machine.direction *= -1;
                    pos.first += 1;
                    if (pos.first >= HEIGHT - BOUNDARY_WIDTH) {
                        pos.first = BOUNDARY_WIDTH;
                        machine.horizontal_sweep = false; // Switch to vertical
                    }
                } else {
                    pos.second = next_col;
                }
            } else {
                // Vertical sweep within territory
                int next_row = pos.first + machine.direction;
                if (next_row >= HEIGHT - BOUNDARY_WIDTH || next_row < BOUNDARY_WIDTH) {
                    // Change column and reverse direction
                    machine.direction *= -1;
                    pos.second += 1;
                    if (pos.second >= machine.territory_end) {
                        pos.second = machine.territory_start;
                        machine.horizontal_sweep = true; // Switch to horizontal
                    }
                } else {
                    pos.first = next_row;
                }
            }
        }
    }

    // Get distance between two positions (Manhattan distance)
    int get_distance(const std::pair<int, int> &pos1, const std::pair<int, int> &pos2) {
        return std::abs(pos1.first - pos2.first) + std::abs(pos1.second - pos2.second);
    }

    // Check if chaser is at same position as a machine
    int chaser_at_machine() {
        for (int i = 0; i < NUM_MACHINES; ++i) {
            if (chaser_position == machines[i].position) {
                return i;
            }
        }
        return -1;
    }

    // Collect from machine if chaser is at machine position
    double collect_from_machine(int machine_id) {
        if (machine_id < 0 || machine_id >= NUM_MACHINES)
            return 0.0;

        int collected = std::min(machines[machine_id].load, CHASER_CAPACITY - chaser_load);
        machines[machine_id].load -= collected;
        chaser_load += collected;

        return collected * COLLECTION_REWARD;
    }

    // Get simplified state for chaser (optimized for faster training)
    int get_chaser_state() {
        // Simplified state representation to reduce state space complexity
        int state = pos_to_state(chaser_position.first, chaser_position.second);

        // Add simplified machine information (just high priority status)
        for (int i = 0; i < NUM_MACHINES; ++i) {
            // High priority flag for machines >80% full
            if (machines[i].load > MACHINE_CAPACITY * 0.8) {
                state += (1 << (10 + i)); // Reduced bit positions
            }
        }

        // Add simplified chaser load status (reduced granularity)
        int load_bucket = std::min(chaser_load / 100, 7); // 8 buckets instead of 16
        state += (load_bucket << 15);                     // Reduced bit position

        return state;
    }

    // Get reward for chaser action with penalty for invalid moves
    double get_chaser_reward(int action, const std::pair<int, int> &old_pos, bool valid_move) {
        double reward = 0.0;

        // Penalty for invalid moves (moving to unharvested cells)
        if (!valid_move) {
            reward += INVALID_MOVE_PENALTY;
        }

        // Movement cost (encourage efficiency)
        if (action != STAY) {
            reward += MOVEMENT_COST;
        }

        // Check if chaser is at a machine and can collect
        int machine_at = chaser_at_machine();
        if (machine_at >= 0 && machines[machine_at].load > 0) {
            reward += collect_from_machine(machine_at);
        }

        // Reward for being close to full machines
        for (int i = 0; i < NUM_MACHINES; ++i) {
            if (machines[i].load > MACHINE_CAPACITY * 0.8) {
                int distance = get_distance(chaser_position, machines[i].position);
                reward += std::max(0.0, 10.0 - distance * 0.02);
            }
        }

        // Heavy penalty for machine overflows
        for (int i = 0; i < NUM_MACHINES; ++i) {
            if (machines[i].load >= MACHINE_CAPACITY) {
                reward += OVERFLOW_PENALTY;
            }
        }

        // Bonus for unloading at boundary (simulating returning to depot)
        if (boundary_cells[chaser_position.first][chaser_position.second] && chaser_load > 0) {
            reward += chaser_load * 0.1; // Small bonus for being at boundary with load
        }

        return reward;
    }

    // Print field visualization (sample area)
    void print_field(int sample_size = 50) {
        std::cout << "\nField Status (" << sample_size << "x" << sample_size << " sample from center):\n";
        std::cout
            << "Legend: . = low density, : = medium, # = high density, B = boundary, M0/M1/M2 = machines, C = chaser\n";

        int start_row = (HEIGHT - sample_size) / 2;
        int start_col = (WIDTH - sample_size) / 2;

        for (int r = start_row; r < start_row + sample_size; ++r) {
            for (int c = start_col; c < start_col + sample_size; ++c) {
                char cell = '.';

                if (boundary_cells[r][c]) {
                    cell = 'B';
                } else {
                    double density = harvest_density[r][c];
                    if (density > 0.7)
                        cell = '#'; // High density
                    else if (density > 0.3)
                        cell = ':'; // Medium density
                    else
                        cell = '.'; // Low/no density
                }

                // Check for agents at this position
                if (std::make_pair(r, c) == chaser_position) {
                    cell = 'C';
                } else {
                    for (int i = 0; i < NUM_MACHINES; ++i) {
                        if (std::make_pair(r, c) == machines[i].position) {
                            cell = '0' + i;
                            break;
                        }
                    }
                }

                std::cout << cell;
            }
            std::cout << "\n";
        }

        std::cout << "\nMachine Status: ";
        for (int i = 0; i < NUM_MACHINES; ++i) {
            std::cout << "M" << i << "=" << machines[i].load << "/" << MACHINE_CAPACITY;
            if (machines[i].harvesting_time_left > 0) {
                std::cout << "(H" << machines[i].harvesting_time_left << ")";
            }
            if (i < NUM_MACHINES - 1)
                std::cout << ", ";
        }
        std::cout << ", Chaser=" << chaser_load << "/" << CHASER_CAPACITY << "\n";
    }

    // Get statistics
    struct FieldStats {
        int total_harvested;
        std::vector<int> machine_loads;
        int chaser_load;
        double harvest_efficiency;
        int total_boundary_cells;
    };

    FieldStats get_stats() {
        double total_density_remaining = 0.0;
        double total_possible_density = 0.0;

        // Calculate remaining harvest density (excluding boundary)
        for (int r = BOUNDARY_WIDTH; r < HEIGHT - BOUNDARY_WIDTH; ++r) {
            for (int c = BOUNDARY_WIDTH; c < WIDTH - BOUNDARY_WIDTH; ++c) {
                total_density_remaining += harvest_density[r][c];
                total_possible_density += 1.0; // Each cell could have had density 1.0
            }
        }

        int total_boundary = 0;
        for (const auto &row : boundary_cells) {
            for (bool cell : row) {
                if (cell)
                    total_boundary++;
            }
        }

        std::vector<int> loads;
        for (const auto &machine : machines) {
            loads.push_back(machine.load);
        }

        // Harvest efficiency = (total possible - remaining) / total possible
        double harvest_efficiency = total_possible_density > 0
                                        ? (total_possible_density - total_density_remaining) / total_possible_density
                                        : 0.0;

        return {static_cast<int>(total_possible_density - total_density_remaining), loads, chaser_load,
                harvest_efficiency, total_boundary};
    }
};

void print_section_header(const std::string &title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

// Run a single episode of the complex harvest simulation
double run_complex_harvest_episode(QLearning<int, int> &chaser_agent, ComplexHarvestField &field, int max_steps = 10000,
                                   bool verbose = false) {
    field.reset();
    double total_reward = 0.0;
    int steps = 0;
    int invalid_moves = 0;

    while (steps < max_steps) {
        // Update machine positions and harvesting
        field.update_machines();

        // Get current state for chaser
        int current_state = field.get_chaser_state();

        // Select action for chaser
        int action = chaser_agent.select_action(current_state);

        // Check if move is valid
        auto old_pos = field.chaser_position;
        auto new_pos = field.get_next_position(field.chaser_position, action);
        bool valid_move = field.can_chaser_move_to(new_pos);

        // Update chaser position (only if valid, otherwise stay)
        if (valid_move) {
            field.chaser_position = new_pos;
        } else {
            invalid_moves++;
            // Keep old position, but still calculate reward with penalty
        }

        // Calculate reward
        double reward = field.get_chaser_reward(action, old_pos, valid_move);
        total_reward += reward;

        // Get next state
        int next_state = field.get_chaser_state();

        // Update Q-learning agent
        chaser_agent.update(current_state, action, reward, next_state, false);

        // Print status periodically
        if (verbose && steps % 1000 == 0) {
            std::cout << "Step " << steps << ", Reward: " << std::fixed << std::setprecision(2) << reward
                      << ", Invalid moves: " << invalid_moves << "\n";
            field.print_field(30);
        }

        steps++;

        // Check termination conditions (more aggressive for faster training)
        auto stats = field.get_stats();
        if (stats.harvest_efficiency > 0.4 || steps > max_steps * 0.8) { // Stop when 40% harvested or 80% of max steps
            if (verbose) {
                std::cout << "Episode complete: " << std::fixed << std::setprecision(1)
                          << stats.harvest_efficiency * 100 << "% of field harvested\n";
            }
            break;
        }
    }

    if (verbose) {
        std::cout << "Episode finished. Total invalid moves: " << invalid_moves << "/" << steps << " (" << std::fixed
                  << std::setprecision(1) << (100.0 * invalid_moves / steps) << "%)\n";
    }

    return total_reward;
}

// Evaluate chaser agent performance
double evaluate_complex_chaser_agent(QLearning<int, int> &agent, int num_episodes = 5) {
    double original_epsilon = agent.get_epsilon();
    agent.set_epsilon(0.0); // No exploration during evaluation

    double total_reward = 0.0;
    for (int i = 0; i < num_episodes; ++i) {
        ComplexHarvestField eval_field;
        total_reward += run_complex_harvest_episode(agent, eval_field, 1500, false); // Reduced evaluation steps
    }

    agent.set_epsilon(original_epsilon); // Restore exploration
    return total_reward / num_episodes;
}

void demo_complex_harvest_coordination() {
    print_section_header("Complex Multi-Agent Harvest Coordination with Penalties");

    ComplexHarvestField field;
    std::vector<int> actions = {0, 1, 2, 3, 4};

    // Create optimized chaser agent with penalty-based learning
    QLearning<int, int> chaser_agent(0.1,  // Increased learning rate for faster convergence
                                     0.95, // Slightly reduced discount factor
                                     0.3,  // Higher exploration initially, will decay
                                     actions, QLearning<int, int>::ExplorationStrategy::BOLTZMANN,
                                     QLearning<int, int>::LearningRateSchedule::EXPONENTIAL_DECAY);

    // Enable optimized advanced features
    chaser_agent.set_double_q_learning(true);
    chaser_agent.set_experience_replay(true, 5000, 128); // Smaller buffer and batch size
    chaser_agent.set_eligibility_traces(true, 0.9);      // Slightly reduced trace decay
    chaser_agent.set_temperature(1.5);                   // Reduced temperature for faster convergence

    // Reward shaping for complex environment
    chaser_agent.set_reward_shaping([](double reward) {
        return std::tanh(reward / 10.0); // Normalize larger reward range
    });

    std::cout << "Training complex chaser agent with:\n";
    std::cout << "  - 100x100 field with 3 variable-speed machines\n";
    std::cout << "  - Penalty-based invalid movement (-5.0 per invalid move)\n";
    std::cout << "  - Boundary cells always accessible\n";
    std::cout << "  - Territorial division with asynchronous operation\n";
    std::cout << "  - Double Q-learning with large experience replay\n";
    std::cout << "  - Eligibility traces and Boltzmann exploration\n\n";

    // Training with optimized parameters for faster convergence
    const int num_episodes = 50; // Reduced episodes for faster training
    double best_performance = -std::numeric_limits<double>::infinity();

    for (int episode = 0; episode < num_episodes; ++episode) {
        double reward = run_complex_harvest_episode(chaser_agent, field, 2000, false); // Reduced steps per episode

        if ((episode + 1) % 10 == 0) {
            double performance = evaluate_complex_chaser_agent(chaser_agent, 3);
            best_performance = std::max(best_performance, performance);

            std::cout << "Episode " << (episode + 1) << ", Performance: " << std::fixed << std::setprecision(1)
                      << performance << ", Best: " << std::fixed << std::setprecision(1) << best_performance << "\n";
        }
    }

    // Print agent statistics
    auto stats = chaser_agent.get_statistics();
    std::cout << "\nComplex Chaser Agent Statistics:\n";
    std::cout << "  Total updates: " << stats.total_updates << "\n";
    std::cout << "  Cumulative reward: " << std::fixed << std::setprecision(1) << stats.cumulative_reward << "\n";
    std::cout << "  Exploration ratio: " << std::fixed << std::setprecision(1) << stats.exploration_ratio * 100
              << "%\n";
    std::cout << "  Q-table size: " << chaser_agent.get_q_table_size() << " states\n";

    // Final demonstration
    std::cout << "\nComplex agent final demonstration:\n";
    run_complex_harvest_episode(chaser_agent, field, 3000, true);
}

int main() {
    std::cout << "Complex Multi-Agent Pea Harvesting Coordination Demo\n";
    std::cout << "====================================================\n";
    std::cout << "Scenario: 3 variable-speed harvesting machines + 1 chaser bin on 100x100 field\n";
    std::cout << "Challenge: Penalty-based learning with territorial division and asynchronous operation\n";

    try {
        demo_complex_harvest_coordination();

        print_section_header("Demo Complete");
        std::cout << "Complex multi-agent harvest coordination successfully demonstrated!\n";
        std::cout << "Key features:\n";
        std::cout << "  ✓ 100x100 field with 3 machines operating at different speeds\n";
        std::cout << "  ✓ Penalty-based movement (-5.0 for invalid moves to unharvested cells)\n";
        std::cout << "  ✓ Boundary cells always accessible for chaser movement\n";
        std::cout << "  ✓ Territorial division preventing machine synchronization\n";
        std::cout << "  ✓ Variable speeds and random initialization for realistic behavior\n";
        std::cout << "  ✓ Advanced Q-learning with large experience replay buffer\n";
        std::cout << "  ✓ Complex state representation with quantized distances\n";

    } catch (const std::exception &e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
