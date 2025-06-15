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

// Add plotting functionality
#include <plotter.hpp>

using namespace relearn::model_free_value_based;

// Complex field environment configuration
struct ComplexHarvestField {
    static constexpr int WIDTH = 100;
    static constexpr int HEIGHT = 100;
    static constexpr int TOTAL_CELLS = WIDTH * HEIGHT;
    static constexpr int BOUNDARY_WIDTH = 2; // Outer boundary always accessible

    // Machine capacity and harvest rates
    static constexpr int MACHINE_CAPACITY = 5000;
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
        bool is_overflowing;         // True if machine has overflowed
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

            // Spawn machines at top or bottom boundary (no crops there)
            // Randomly choose top (row 0-1) or bottom (row 98-99)
            bool spawn_at_top = (rng() % 2 == 0);
            int spawn_row =
                spawn_at_top ? (rng() % BOUNDARY_WIDTH) : (HEIGHT - BOUNDARY_WIDTH + (rng() % BOUNDARY_WIDTH));
            std::uniform_int_distribution<> col_dist(machine.territory_start, machine.territory_end - 1);

            machine.position = {spawn_row, col_dist(rng)};
            machine.load = 0;
            machine.speed_multiplier = 1.0;           // All machines have identical speed
            machine.harvest_rate = BASE_HARVEST_RATE; // All machines have same harvest rate
            machine.direction = (rng() % 2 == 0) ? 1 : -1;
            machine.horizontal_sweep = (rng() % 2 == 0);
            machine.steps_since_move = 0;
            machine.move_interval = 1; // All machines move every step when not harvesting
            machine.harvesting_time_left = 0;
            machine.current_cell_density = 0.0;
            machine.is_overflowing = false; // Initialize overflow flag

            std::cout << "Machine " << i << " initialized:\n";
            std::cout << "  Territory: cols " << machine.territory_start << "-" << machine.territory_end << "\n";
            std::cout << "  Starting position: (" << machine.position.first << ", " << machine.position.second << ")\n";
            std::cout << "  Movement pattern: " << (machine.horizontal_sweep ? "Horizontal" : "Vertical")
                      << " sweep\n\n";
        }

        // Spawn chaser bin at boundary area (top, bottom, left, or right edge)
        // Randomly choose which boundary: 0=top, 1=bottom, 2=left, 3=right
        int boundary_choice = rng() % 4;
        int chaser_row, chaser_col;

        if (boundary_choice == 0) {
            // Top boundary
            chaser_row = rng() % BOUNDARY_WIDTH;
            chaser_col = rng() % WIDTH;
        } else if (boundary_choice == 1) {
            // Bottom boundary
            chaser_row = HEIGHT - BOUNDARY_WIDTH + (rng() % BOUNDARY_WIDTH);
            chaser_col = rng() % WIDTH;
        } else if (boundary_choice == 2) {
            // Left boundary
            chaser_row = rng() % HEIGHT;
            chaser_col = rng() % BOUNDARY_WIDTH;
        } else {
            // Right boundary
            chaser_row = rng() % HEIGHT;
            chaser_col = WIDTH - BOUNDARY_WIDTH + (rng() % BOUNDARY_WIDTH);
        }

        chaser_position = {chaser_row, chaser_col};

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

    // Check if chaser can move to a position (boundary OR harvested cells OR current position)
    bool can_chaser_move_to(const std::pair<int, int> &pos) {
        if (!is_valid_position(pos))
            return false;
        return boundary_cells[pos.first][pos.second] ||
               harvest_density[pos.first][pos.second] == 0.0 || // Only completely harvested cells
               pos == chaser_position;
    }

    // Update machine positions and harvest with step-by-step harvesting
    void update_machines() {
        for (int i = 0; i < NUM_MACHINES; ++i) {
            Machine &machine = machines[i];
            auto &pos = machine.position;

            // Harvest current cell gradually (0.2 per step)
            if (harvest_density[pos.first][pos.second] > 0.01) {
                const double HARVEST_PER_STEP = 0.2;

                // Harvest from current cell (only if not overflowing)
                if (!machine.is_overflowing) {
                    double harvest_amount = std::min(HARVEST_PER_STEP, harvest_density[pos.first][pos.second]);
                    harvest_density[pos.first][pos.second] -= harvest_amount;

                    // Add to machine load proportional to what was harvested
                    int load_increase = static_cast<int>(harvest_amount * 10); // Scale: 10 load units per 1.0 density
                    int old_load = machine.load;
                    machine.load = std::min(MACHINE_CAPACITY, machine.load + load_increase);

                    // Check for overflow
                    if (machine.load >= MACHINE_CAPACITY && !machine.is_overflowing) {
                        machine.is_overflowing = true;
                        std::cout << "WARNING: Machine " << i << " is now overflowing! (Load: " << machine.load << "/"
                                  << MACHINE_CAPACITY << ")\n";
                    }
                }

                // Stay on this cell until it's completely harvested (don't move yet)
                continue;
            }

            // Current cell is harvested (density <= 0.01), move to next position
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
            int distance = get_distance(chaser_position, machines[i].position);
            if (distance <= 3) { // Within 3 cells radius
                return i;
            }
        }
        return -1;
    }

    // Collect from machine if chaser is at machine position
    double collect_from_machine(int machine_id) {
        if (machine_id < 0 || machine_id >= NUM_MACHINES)
            return 0.0;

        Machine &machine = machines[machine_id];
        if (machine.load == 0)
            return 0.0; // No load to collect

        // Check if this is a good or bad unloading decision
        double capacity_ratio = (double)machine.load / MACHINE_CAPACITY;
        double unloading_reward = 0.0;

        if (capacity_ratio >= 0.8) {
            // Good unloading: machine is ≥80% full
            unloading_reward += 50.0; // Large positive reward
        } else {
            // Bad unloading: machine is <80% full
            unloading_reward -= 5.0; // Penalty for premature unloading
        }

        int collected = std::min(machine.load, CHASER_CAPACITY - chaser_load);
        machine.load -= collected;
        chaser_load += collected;

        // Reset overflow flag if machine is no longer at capacity
        if (machine.load < MACHINE_CAPACITY) {
            machine.is_overflowing = false;
        }

        return collected * COLLECTION_REWARD + unloading_reward;
    }

    // Get simplified state for chaser (optimized for faster training)
    int get_chaser_state() {
        // Enhanced state representation with machine positions and fill levels
        int state = 0;

        // 1. Chaser position (discretized to reduce state space)
        int chaser_x_bucket = chaser_position.first / 10;  // 10 buckets for 100x100 field
        int chaser_y_bucket = chaser_position.second / 10; // 10 buckets for 100x100 field
        state += chaser_x_bucket * 1000 + chaser_y_bucket * 100;

        // 2. For each machine: position and fill level
        for (int i = 0; i < NUM_MACHINES; ++i) {
            // Machine position (discretized)
            int machine_x_bucket = machines[i].position.first / 20;  // 5 buckets for position
            int machine_y_bucket = machines[i].position.second / 20; // 5 buckets for position

            // Machine fill level (5 categories: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
            int fill_level = std::min(4, machines[i].load * 5 / MACHINE_CAPACITY);

            // Combine machine info: position + fill level
            int machine_info = machine_x_bucket * 125 + machine_y_bucket * 25 + fill_level * 5;
            state += machine_info * (int)std::pow(10000, i); // Separate bit ranges for each machine
        }

        // 3. Chaser load status (discretized)
        int chaser_load_bucket = std::min(4, chaser_load * 5 / CHASER_CAPACITY);
        state += chaser_load_bucket; // Add to lowest bits

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

    // ============ PLOTTING FUNCTIONS ============

    /// Plot the current field state with machines and chaser
    void plot_field(plotter::Plotter &plt, const std::string &title = "Harvest Field", int sample_size = 100) {
        plt.clf(); // Clear previous plot

        // Set up the plot
        plt.figure_size(1000, 1000);
        plt.xlim(-5, sample_size + 5);
        plt.ylim(-5, sample_size + 5);
        plt.title(title.c_str());

        // Calculate sample area (center of field)
        int start_row = (HEIGHT - sample_size) / 2;
        int start_col = (WIDTH - sample_size) / 2;

        // Plot harvest density as colored rectangles
        for (int r = 0; r < sample_size; ++r) {
            for (int c = 0; c < sample_size; ++c) {
                int field_r = start_row + r;
                int field_c = start_col + c;

                if (field_r >= 0 && field_r < HEIGHT && field_c >= 0 && field_c < WIDTH) {
                    pigment::RGB color;

                    if (boundary_cells[field_r][field_c]) {
                        // Boundary cells - gray
                        color = pigment::RGB(180, 180, 180);
                    } else {
                        // Crop density - green gradient, WHITE if harvested (density = 0)
                        double density = harvest_density[field_r][field_c];
                        if (density <= 0.01) {
                            // Harvested cell - WHITE
                            color = pigment::RGB(255, 255, 255);
                        } else {
                            // Remaining crop - green gradient
                            int green_value = 80 + static_cast<int>(density * 175); // 80 to 255
                            color = pigment::RGB(50, green_value, 50);              // Dark to bright green
                        }
                    }

                    // Fill small rectangle for this cell
                    auto rect_points =
                        plotter::integrations::geometry::generate_rectangle_points(c, sample_size - r - 1, 1, 1);
                    plt.fill(rect_points, color);
                }
            }
        }

        // Plot machines as colored circles
        std::vector<pigment::RGB> machine_colors = {
            pigment::RGB(255, 50, 50), // Red for Machine 0
            pigment::RGB(50, 50, 255), // Blue for Machine 1
            pigment::RGB(255, 150, 0)  // Orange for Machine 2
        };

        for (int i = 0; i < NUM_MACHINES; ++i) {
            int machine_r = machines[i].position.first;
            int machine_c = machines[i].position.second;

            // Check if machine is in our sample area
            if (machine_r >= start_row && machine_r < start_row + sample_size && machine_c >= start_col &&
                machine_c < start_col + sample_size) {

                double plot_x = machine_c - start_col;
                double plot_y = sample_size - (machine_r - start_row) - 1;

                // Circle size based on load
                double radius = 2.0 + (machines[i].load / 100.0) * 3.0; // 2-5 units radius
                plt.plot_circle(plot_x + 0.5, plot_y + 0.5, radius, machine_colors[i]);
            }
        }

        // Plot chaser as yellow star (using diamond shape)
        if (chaser_position.first >= start_row && chaser_position.first < start_row + sample_size &&
            chaser_position.second >= start_col && chaser_position.second < start_col + sample_size) {

            double plot_x = chaser_position.second - start_col;
            double plot_y = sample_size - (chaser_position.first - start_row) - 1;

            // Create diamond shape for chaser
            std::vector<concord::Point> diamond_points;
            diamond_points.emplace_back(plot_x + 0.5, plot_y + 2.5); // Top
            diamond_points.emplace_back(plot_x + 2.5, plot_y + 0.5); // Right
            diamond_points.emplace_back(plot_x + 0.5, plot_y - 1.5); // Bottom
            diamond_points.emplace_back(plot_x - 1.5, plot_y + 0.5); // Left
            diamond_points.emplace_back(plot_x + 0.5, plot_y + 2.5); // Close shape

            // Yellow chaser with size based on load
            double load_factor = 1.0 + (chaser_load / 500.0); // Scale 1.0-2.0
            for (auto &point : diamond_points) {
                point = concord::Point(plot_x + 0.5 + (point.x - plot_x - 0.5) * load_factor,
                                       plot_y + 0.5 + (point.y - plot_y - 0.5) * load_factor);
            }

            plt.fill(diamond_points, pigment::RGB(255, 255, 0)); // Yellow
        }

        // Add machine status text in title
        auto stats = get_stats();
        std::string full_title = title + " (Efficiency: " + std::to_string(int(stats.harvest_efficiency * 100)) +
                                 "%, M0:" + std::to_string(machines[0].load) +
                                 " M1:" + std::to_string(machines[1].load) + " M2:" + std::to_string(machines[2].load) +
                                 " C:" + std::to_string(chaser_load) + ")";
        plt.title(full_title.c_str());
    }

    /// Create simplified animated visualization for better performance
    void plot_learning_progress(plotter::Plotter &plt, int episode, double performance, double best_performance,
                                int sample_size = 100, double current_reward = 0.0) {
        plt.clf(); // Clear previous plot

        // Set up the plot with extra space on the right for capacity bars
        plt.figure_size(1200, 1000);
        plt.xlim(-5, sample_size + 15); // Extra space on right for capacity bars
        plt.ylim(-5, sample_size + 5);

        // Create title with key info
        std::string title = "Episode " + std::to_string(episode) +
                            " - Performance: " + std::to_string(performance).substr(0, 4) +
                            " (Best: " + std::to_string(best_performance).substr(0, 4) + ")";
        plt.title(title.c_str());

        // Calculate sample area (center of field)
        int start_row = (HEIGHT - sample_size) / 2;
        int start_col = (WIDTH - sample_size) / 2;

        // Plot only key elements for animation (much faster)

        // 1. Plot field background - use full resolution for consistency
        for (int r = 0; r < sample_size; ++r) {
            for (int c = 0; c < sample_size; ++c) {
                int field_r = start_row + r;
                int field_c = start_col + c;

                if (field_r >= 0 && field_r < HEIGHT && field_c >= 0 && field_c < WIDTH) {
                    pigment::RGB color;

                    if (boundary_cells[field_r][field_c]) {
                        // Boundary cells - light gray
                        color = pigment::RGB(180, 180, 180);
                    } else {
                        // Crop density - green gradient, WHITE if harvested (density = 0)
                        double density = harvest_density[field_r][field_c];
                        if (density <= 0.01) {
                            // Harvested cell - WHITE
                            color = pigment::RGB(255, 255, 255);
                        } else {
                            // Remaining crop - green gradient
                            int green_value = 80 + static_cast<int>(density * 175); // 80 to 255
                            color = pigment::RGB(50, green_value, 50);              // Dark to bright green
                        }
                    }

                    // Check if there's a machine or chaser at this position - override with bright colors
                    bool has_agent = false;

                    // Check for machines
                    for (int i = 0; i < NUM_MACHINES; ++i) {
                        if (machines[i].position.first == field_r && machines[i].position.second == field_c) {
                            // Machine colors: bright red, bright blue, bright orange
                            if (i == 0)
                                color = pigment::RGB(255, 0, 0); // Bright red
                            else if (i == 1)
                                color = pigment::RGB(0, 0, 255); // Bright blue
                            else
                                color = pigment::RGB(255, 165, 0); // Bright orange
                            has_agent = true;
                            break;
                        }
                    }

                    // Check for chaser
                    if (!has_agent && chaser_position.first == field_r && chaser_position.second == field_c) {
                        color = pigment::RGB(255, 20, 147); // Bright pink/magenta for chaser
                        has_agent = true;
                    }

                    // Fill rectangle for this cell
                    auto rect_points =
                        plotter::integrations::geometry::generate_rectangle_points(c, sample_size - r - 1, 1, 1);
                    plt.fill(rect_points, color);
                }
            }
        }

        // 2. Plot territorial boundaries as separate vertical lines
        int territory_width = (WIDTH - 2 * BOUNDARY_WIDTH) / NUM_MACHINES;
        for (int i = 1; i < NUM_MACHINES; ++i) {
            int boundary_col = BOUNDARY_WIDTH + i * territory_width;
            if (boundary_col >= start_col && boundary_col < start_col + sample_size) {
                int plot_x = boundary_col - start_col;
                // Draw each boundary as a separate vertical line
                std::vector<concord::Point> single_boundary_line;
                single_boundary_line.emplace_back(plot_x, -5);
                single_boundary_line.emplace_back(plot_x, sample_size + 5);
                plt.plot(single_boundary_line, pigment::RGB(100, 100, 100)); // Gray territory line
            }
        }

        // 3. Add machine capacity visualization (bars on the right side)
        std::vector<pigment::RGB> machine_colors = {
            pigment::RGB(255, 0, 0),  // Red for Machine 0
            pigment::RGB(0, 0, 255),  // Blue for Machine 1
            pigment::RGB(255, 165, 0) // Orange for Machine 2
        };

        // Draw capacity bars for each machine on the right side
        for (int i = 0; i < NUM_MACHINES; ++i) {
            double capacity_ratio = (double)machines[i].load / MACHINE_CAPACITY;
            double bar_height = capacity_ratio * 20; // Max 20 units tall
            double bar_x = sample_size + 2;
            double bar_y = 20 + i * 25; // Spacing between bars

            // Background bar (empty capacity)
            auto bg_rect = plotter::integrations::geometry::generate_rectangle_points(bar_x, bar_y, 3, 20);
            plt.fill(bg_rect, pigment::RGB(200, 200, 200)); // Light gray background

            // Filled portion (current load) - use different colors based on capacity
            if (bar_height > 0) {
                pigment::RGB bar_color;
                if (capacity_ratio >= 1.0) {
                    bar_color = pigment::RGB(255, 0, 0); // Red for overflow
                } else if (capacity_ratio >= 0.8) {
                    bar_color = pigment::RGB(255, 165, 0); // Orange for ≥80%
                } else {
                    bar_color = machine_colors[i]; // Normal machine color
                }
                auto fill_rect =
                    plotter::integrations::geometry::generate_rectangle_points(bar_x, bar_y, 3, bar_height);
                plt.fill(fill_rect, bar_color);
            }

            // Add text label showing percentage
            std::string label = "M" + std::to_string(i) + ":" + std::to_string(int(capacity_ratio * 100)) + "%";
            // Note: Text positioning might need adjustment based on plotter API
        }

        // 4. Add complete status information (legend with machine capacities and reward)
        auto stats = get_stats();

        // Count overflowing machines
        int overflowing_count = 0;
        for (int i = 0; i < NUM_MACHINES; ++i) {
            if (machines[i].is_overflowing)
                overflowing_count++;
        }

        std::string overflow_status = "";
        if (overflowing_count > 0) {
            overflow_status = " | OVERFLOW:" + std::to_string(overflowing_count) + "/" + std::to_string(NUM_MACHINES);
        }

        std::string status = "Efficiency: " + std::to_string(int(stats.harvest_efficiency * 100)) +
                             "% | Reward: " + std::to_string(int(current_reward)) +
                             " | M0:" + std::to_string(machines[0].load) + "/" + std::to_string(MACHINE_CAPACITY) +
                             " M1:" + std::to_string(machines[1].load) + "/" + std::to_string(MACHINE_CAPACITY) +
                             " M2:" + std::to_string(machines[2].load) + "/" + std::to_string(MACHINE_CAPACITY) +
                             overflow_status;
        plt.xlabel(status.c_str());
    }

    // Debug function to show chaser's knowledge of machine states
    void print_chaser_knowledge() {
        std::cout << "Chaser Knowledge:\n";
        std::cout << "  Chaser position: (" << chaser_position.first << ", " << chaser_position.second << ")\n";
        std::cout << "  Chaser load: " << chaser_load << "/" << CHASER_CAPACITY << "\n";

        for (int i = 0; i < NUM_MACHINES; ++i) {
            double fill_percentage = (double)machines[i].load / MACHINE_CAPACITY * 100;
            int distance = get_distance(chaser_position, machines[i].position);

            std::cout << "  Machine " << i << ": pos(" << machines[i].position.first << ","
                      << machines[i].position.second << ") load=" << machines[i].load << "/" << MACHINE_CAPACITY << " ("
                      << std::fixed << std::setprecision(1) << fill_percentage << "%) distance=" << distance << "\n";
        }
    }

    // ...existing code...
};

void print_section_header(const std::string &title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

// Run a single episode with detailed step-by-step visualization
double run_detailed_visualization_episode(QLearning<int, int> &chaser_agent, ComplexHarvestField &field,
                                          plotter::Plotter &plt, int max_steps = 500) {
    field.reset();
    double total_reward = 0.0;
    int steps = 0;
    int invalid_moves = 0;

    std::cout << "Starting detailed episode visualization...\n";
    std::cout << "Initial positions:\n";
    for (int i = 0; i < field.NUM_MACHINES; ++i) {
        std::cout << "  Machine " << i << ": (" << field.machines[i].position.first << ", "
                  << field.machines[i].position.second << ")\n";
    }
    std::cout << "  Chaser: (" << field.chaser_position.first << ", " << field.chaser_position.second << ")\n\n";

    // Capture initial frame
    field.plot_learning_progress(plt, 0, 0.0, 0.0, 100, 0.0);
    plt.frame();
    std::cout << "Frame " << plt.frame_count() << " captured (initial state)\n";

    while (max_steps < 0 || steps < max_steps) {
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
        }

        // Calculate reward
        double reward = field.get_chaser_reward(action, old_pos, valid_move);
        total_reward += reward;

        // Get next state
        int next_state = field.get_chaser_state();

        // Update Q-learning agent
        chaser_agent.update(current_state, action, reward, next_state, false);

        steps++;

        // Capture frame every 50 steps for detailed visualization
        if (steps % 50 == 0) {
            auto stats = field.get_stats();
            field.plot_learning_progress(plt, steps, stats.harvest_efficiency, stats.harvest_efficiency, 100,
                                         total_reward);
            plt.frame();
            std::cout << "Frame " << plt.frame_count() << " captured (step " << steps << ", efficiency: " << std::fixed
                      << std::setprecision(1) << stats.harvest_efficiency * 100 << "%)\n";
        }

        // Print status every 50 steps
        if (steps % 50 == 0) {
            auto stats = field.get_stats();
            std::cout << "Step " << steps << " - Efficiency: " << std::fixed << std::setprecision(1)
                      << stats.harvest_efficiency * 100 << "%, Invalid moves: " << invalid_moves << "\n";

            // Show current positions
            std::cout << "  Positions: M0(" << field.machines[0].position.first << ","
                      << field.machines[0].position.second << ") M1(" << field.machines[1].position.first << ","
                      << field.machines[1].position.second << ") M2(" << field.machines[2].position.first << ","
                      << field.machines[2].position.second << ") C(" << field.chaser_position.first << ","
                      << field.chaser_position.second << ")\n";

            // Show chaser's knowledge of machine states
            field.print_chaser_knowledge();
        }

        // Check termination conditions
        auto stats = field.get_stats();

        // Hybrid overflow approach: penalties but continue, terminate only if all machines overflow
        int overflowing_machines = 0;
        double overflow_penalty = 0.0;

        for (int i = 0; i < ComplexHarvestField::NUM_MACHINES; ++i) {
            if (field.machines[i].is_overflowing) {
                overflowing_machines++;
                if (overflowing_machines == 1) {
                    overflow_penalty -= 500.0; // First overflow: -500 per step
                } else {
                    overflow_penalty -= 1000.0; // Multiple overflows: -1000 per step
                }
            }
        }

        total_reward += overflow_penalty;

        // Terminate only if ALL machines are overflowing (complete failure)
        bool complete_failure = (overflowing_machines >= ComplexHarvestField::NUM_MACHINES);
        bool harvest_complete = (stats.harvest_efficiency > 0.6);
        bool max_steps_reached = (max_steps > 0 && steps >= max_steps);

        // Check for termination and print clear reason
        if (complete_failure || harvest_complete || max_steps_reached) {
            std::cout << "\n" << std::string(60, '=') << "\n";
            std::cout << "EPISODE TERMINATED at step " << steps << "\n";
            std::cout << std::string(60, '=') << "\n";

            if (complete_failure) {
                std::cout << "REASON: All " << ComplexHarvestField::NUM_MACHINES
                          << " machines are overflowing (complete failure)\n";
                for (int i = 0; i < ComplexHarvestField::NUM_MACHINES; ++i) {
                    std::cout << "  Machine " << i << ": " << field.machines[i].load << "/"
                              << ComplexHarvestField::MACHINE_CAPACITY << " (" << std::fixed << std::setprecision(1)
                              << (double)field.machines[i].load / ComplexHarvestField::MACHINE_CAPACITY * 100
                              << "%) - OVERFLOWING\n";
                }
            } else if (harvest_complete) {
                std::cout << "REASON: Harvest efficiency target reached (" << std::fixed << std::setprecision(1)
                          << stats.harvest_efficiency * 100 << "% >= 60%)\n";
                std::cout << "SUCCESS: Field successfully harvested!\n";
            } else if (max_steps_reached) {
                std::cout << "REASON: Maximum steps reached (" << steps << " >= " << max_steps << ")\n";
                std::cout << "TIMEOUT: Episode ran too long\n";
            }

            // Show final machine status
            std::cout << "\nFinal machine status:\n";
            for (int i = 0; i < ComplexHarvestField::NUM_MACHINES; ++i) {
                double capacity_percent = (double)field.machines[i].load / ComplexHarvestField::MACHINE_CAPACITY * 100;
                std::string status_flag = field.machines[i].is_overflowing ? " [OVERFLOW]" : "";
                std::cout << "  Machine " << i << ": " << field.machines[i].load << "/"
                          << ComplexHarvestField::MACHINE_CAPACITY << " (" << std::fixed << std::setprecision(1)
                          << capacity_percent << "%)" << status_flag << "\n";
            }

            std::cout << "  Chaser: " << field.chaser_load << "/" << ComplexHarvestField::CHASER_CAPACITY << " ("
                      << std::fixed << std::setprecision(1)
                      << (double)field.chaser_load / ComplexHarvestField::CHASER_CAPACITY * 100 << "%)\n";

            std::cout << std::string(60, '=') << "\n";

            break;
        }
    }

    // Capture final frame
    auto final_stats = field.get_stats();
    field.plot_learning_progress(plt, steps, final_stats.harvest_efficiency, final_stats.harvest_efficiency, 100,
                                 total_reward);
    plt.frame();
    std::cout << "Frame " << plt.frame_count() << " captured (final state)\n";

    std::cout << "Episode finished. Total invalid moves: " << invalid_moves << "/" << steps << " (" << std::fixed
              << std::setprecision(1) << (100.0 * invalid_moves / steps) << "%)\n";

    return total_reward;
}
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

    // ============ PLOTTING SETUP ============
    plotter::Plotter plt;

    std::cout << "\nCreating detailed step-by-step visualization of one episode...\n";
    std::cout << "This will show machine movement pixel by pixel and harvested areas turning white.\n";

    // Enable animation mode for GIF creation (shorter interval for smoother animation)
    plt.enable_animation(200); // 200ms per frame for detailed visualization

    // Clear any existing frames from previous runs
    plt.clear_frames();

    std::cout << "Running detailed episode with:\n";
    std::cout << "  - 100x100 field with 3 machines\n";
    std::cout << "  - Step-by-step movement visualization\n";
    std::cout << "  - Harvested cells turn white\n";
    std::cout << "  - Frame captured every 10 steps\n";
    std::cout << "  - Real-time position tracking\n\n";

    // Run one detailed episode instead of training loop (shorter to avoid too many frames)
    double episode_reward = run_detailed_visualization_episode(chaser_agent, field, plt, -1); // No step limit

    // Print episode statistics
    std::cout << "\nDetailed Episode Statistics:\n";
    std::cout << "  Episode reward: " << std::fixed << std::setprecision(1) << episode_reward << "\n";
    auto final_stats = field.get_stats();
    std::cout << "  Final efficiency: " << std::fixed << std::setprecision(1) << final_stats.harvest_efficiency * 100
              << "%\n";
    std::cout << "  Machine loads: M0=" << field.machines[0].load << " M1=" << field.machines[1].load
              << " M2=" << field.machines[2].load << "\n";
    std::cout << "  Chaser load: " << field.chaser_load << "\n";
    std::cout << "  Total animation frames: " << plt.frame_count() << "\n";

    // ============ SAVE FINAL VISUALIZATION ============

    // Plot final state with full detail
    field.plot_field(plt, "Final State - Detailed Episode Visualization", 100);

    std::cout << "\nSaving visualizations...\n";

    // Save final static image
    plt.save("harvest_demo_final.png", false);
    std::cout << "  Static final image saved as: harvest_demo_final.png\n";

    // Save animated GIF of detailed episode
    if (plt.is_animation_enabled() && plt.frame_count() > 0) {
        plt.save("harvest_demo_detailed_episode.gif", true);
        std::cout << "  Detailed episode animation saved as: harvest_demo_detailed_episode.gif\n";
        std::cout << "  Animation contains " << plt.frame_count() << " frames\n";
    }

    std::cout << "\nVisualization complete! Check the generated files:\n";
    std::cout << "  🎯 harvest_demo_final.png - Final state of detailed episode\n";
    std::cout << "  🎬 harvest_demo_detailed_episode.gif - Complete episode animation\n";
    std::cout << "     Shows pixel-by-pixel machine movement and harvested areas turning white\n";
}

int main() {
    std::cout << "Complex Multi-Agent Pea Harvesting Coordination Demo\n";
    std::cout << "====================================================\n";
    std::cout << "Scenario: Detailed step-by-step visualization of one harvest episode\n";
    std::cout << "Challenge: Track pixel-perfect machine movement and harvesting progress\n";

    try {
        demo_complex_harvest_coordination();

        print_section_header("Demo Complete");
        std::cout << "Detailed episode visualization successfully completed!\n";
        std::cout << "Key features demonstrated:\n";
        std::cout << "  ✓ Pixel-perfect machine movement tracking\n";
        std::cout << "  ✓ Harvested cells turning white when empty\n";
        std::cout << "  ✓ Frame-by-frame progression (every 10 steps)\n";
        std::cout << "  ✓ Territorial boundaries clearly visible\n";
        std::cout << "  ✓ Real-time position and efficiency monitoring\n";
        std::cout << "  ✓ Complete episode from start to finish\n";
        std::cout << "  ✓ Detailed animation export to GIF\n";

    } catch (const std::exception &e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
