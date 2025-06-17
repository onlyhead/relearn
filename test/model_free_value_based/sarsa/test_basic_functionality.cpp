#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>

// Include the SARSA implementation
#include "../../../include/relearn/model_free_value_based/sarsa.hpp"

using namespace relearn::model_free_value_based;

// Simple test environment - Grid World
struct GridState {
    int x, y;
    
    GridState(int x = 0, int y = 0) : x(x), y(y) {}
    
    bool operator==(const GridState& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator<(const GridState& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
};

// Hash function for GridState
namespace std {
    template<>
    struct hash<GridState> {
        size_t operator()(const GridState& state) const {
            return hash<int>()(state.x) ^ (hash<int>()(state.y) << 1);
        }
    };
}

// Stream operators for GridState (needed for save/load)
std::ostream& operator<<(std::ostream& os, const GridState& state) {
    os << state.x << "," << state.y;
    return os;
}

std::istream& operator>>(std::istream& is, GridState& state) {
    char comma;
    is >> state.x >> comma >> state.y;
    return is;
}

enum class Action { UP, DOWN, LEFT, RIGHT };

// Stream operators for Action (needed for save/load)
std::ostream& operator<<(std::ostream& os, const Action& action) {
    os << static_cast<int>(action);
    return os;
}

std::istream& operator>>(std::istream& is, Action& action) {
    int val;
    is >> val;
    action = static_cast<Action>(val);
    return is;
}

class GridWorld {
private:
    int width_, height_;
    GridState goal_;
    std::vector<GridState> obstacles_;
    std::mt19937 rng_;
    
public:
    GridWorld(int width, int height, GridState goal, std::vector<GridState> obstacles = {})
        : width_(width), height_(height), goal_(goal), obstacles_(obstacles), rng_(std::random_device{}()) {}
    
    std::pair<GridState, double> step(const GridState& state, Action action) {
        GridState next_state = state;
        
        // Apply action
        switch (action) {
            case Action::UP:    next_state.y = std::max(0, state.y - 1); break;
            case Action::DOWN:  next_state.y = std::min(height_ - 1, state.y + 1); break;
            case Action::LEFT:  next_state.x = std::max(0, state.x - 1); break;
            case Action::RIGHT: next_state.x = std::min(width_ - 1, state.x + 1); break;
        }
        
        // Check for obstacles
        for (const auto& obstacle : obstacles_) {
            if (next_state == obstacle) {
                next_state = state; // Stay in place if hitting obstacle
                break;
            }
        }
        
        // Calculate reward
        double reward = -1.0; // Living cost
        if (next_state == goal_) {
            reward = 10.0; // Goal reward
        }
        
        return {next_state, reward};
    }
    
    bool is_terminal(const GridState& state) const {
        return state == goal_;
    }
    
    GridState get_start_state() const {
        return GridState(0, 0);
    }
    
    std::vector<Action> get_valid_actions(const GridState& state) const {
        std::vector<Action> actions;
        
        if (state.y > 0) actions.push_back(Action::UP);
        if (state.y < height_ - 1) actions.push_back(Action::DOWN);
        if (state.x > 0) actions.push_back(Action::LEFT);
        if (state.x < width_ - 1) actions.push_back(Action::RIGHT);
        
        return actions;
    }
};

// Test runner class
class SARSABasicTest {
private:
    bool verbose_;
    int passed_tests_;
    int total_tests_;
    
    void assert_test(bool condition, const std::string& test_name) {
        total_tests_++;
        if (condition) {
            passed_tests_++;
            if (verbose_) {
                std::cout << "✓ " << test_name << " PASSED" << std::endl;
            }
        } else {
            std::cout << "✗ " << test_name << " FAILED" << std::endl;
        }
        assert(condition);
    }
    
public:
    SARSABasicTest(bool verbose = false) : verbose_(verbose), passed_tests_(0), total_tests_(0) {}
    
    void test_constructor() {
        if (verbose_) std::cout << "\n=== Testing Constructor ===" << std::endl;
        
        // Test default constructor
        SARSA<GridState, Action> sarsa1;
        assert_test(sarsa1.get_alpha() == 0.1, "Default alpha");
        assert_test(sarsa1.get_gamma() == 0.95, "Default gamma");
        assert_test(sarsa1.get_epsilon() == 0.1, "Default epsilon");
        
        // Test custom parameters
        SARSA<GridState, Action> sarsa2(0.2, 0.9, 0.05);
        assert_test(sarsa2.get_alpha() == 0.2, "Custom alpha");
        assert_test(sarsa2.get_gamma() == 0.9, "Custom gamma");
        assert_test(sarsa2.get_epsilon() == 0.05, "Custom epsilon");
        
        // Test with actions
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa3(0.1, 0.95, 0.1, actions);
        assert_test(true, "Constructor with actions");
    }
    
    void test_parameter_setting() {
        if (verbose_) std::cout << "\n=== Testing Parameter Setting ===" << std::endl;
        
        SARSA<GridState, Action> sarsa;
        
        // Test alpha setting
        sarsa.set_alpha(0.5);
        assert_test(sarsa.get_alpha() == 0.5, "Set alpha");
        
        // Test gamma setting
        sarsa.set_gamma(0.8);
        assert_test(sarsa.get_gamma() == 0.8, "Set gamma");
        
        // Test epsilon setting
        sarsa.set_epsilon(0.2);
        assert_test(sarsa.get_epsilon() == 0.2, "Set epsilon");
        
        // Test lambda setting
        sarsa.set_lambda(0.7);
        assert_test(sarsa.get_lambda() == 0.7, "Set lambda");
        
        // Test boundary constraints
        sarsa.set_gamma(1.5); // Should be clamped to 1.0
        assert_test(sarsa.get_gamma() == 1.0, "Gamma upper bound");
        
        sarsa.set_gamma(-0.1); // Should be clamped to 0.0
        assert_test(sarsa.get_gamma() == 0.0, "Gamma lower bound");
    }
    
    void test_q_value_operations() {
        if (verbose_) std::cout << "\n=== Testing Q-Value Operations ===" << std::endl;
        
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
        sarsa.set_actions(actions);
        
        GridState state(1, 1);
        
        // Test initial Q-values (should be 0)
        assert_test(sarsa.get_q_value(state, Action::UP) == 0.0, "Initial Q-value");
        
        // Test basic update
        GridState next_state(1, 0);
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        
        // Q-value should have changed
        double q_after_update = sarsa.get_q_value(state, Action::UP);
        assert_test(q_after_update != 0.0, "Q-value changed after update");
        
        // Test multiple updates
        double q_before = sarsa.get_q_value(state, Action::UP);
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        double q_after = sarsa.get_q_value(state, Action::UP);
        assert_test(q_after != q_before, "Q-value changed after second update");
    }
    
    void test_action_selection() {
        if (verbose_) std::cout << "\n=== Testing Action Selection ===" << std::endl;
        
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.0, actions); // No exploration
        sarsa.set_actions(actions);
        
        GridState state(1, 1);
        
        // With zero exploration, should always select same action initially
        Action first_action = sarsa.select_action(state);
        Action second_action = sarsa.select_action(state);
        assert_test(first_action == second_action, "Consistent action selection with zero exploration");
        
        // Test with high exploration
        sarsa.set_epsilon(1.0); // Always explore
        bool different_actions = false;
        Action base_action = sarsa.select_action(state);
        
        for (int i = 0; i < 20; ++i) {
            if (sarsa.select_action(state) != base_action) {
                different_actions = true;
                break;
            }
        }
        assert_test(different_actions, "Different actions with full exploration");
    }
    
    void test_best_action() {
        if (verbose_) std::cout << "\n=== Testing Best Action ===" << std::endl;
        
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
        sarsa.set_actions(actions);
        
        GridState state(1, 1);
        
        // Manually set Q-values to test best action selection
        sarsa.update(state, Action::UP, 10.0, GridState(1, 0), Action::RIGHT, false);
        sarsa.update(state, Action::DOWN, 5.0, GridState(1, 2), Action::RIGHT, false);
        sarsa.update(state, Action::LEFT, 2.0, GridState(0, 1), Action::RIGHT, false);
        sarsa.update(state, Action::RIGHT, 1.0, GridState(2, 1), Action::RIGHT, false);
        
        Action best = sarsa.get_best_action(state);
        assert_test(best == Action::UP, "Best action selection");
    }
    
    void test_terminal_state_handling() {
        if (verbose_) std::cout << "\n=== Testing Terminal State Handling ===" << std::endl;
        
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.5, 0.9, 0.1, actions); // High learning rate for visible effect
        sarsa.set_actions(actions);
        
        GridState state(2, 2);
        GridState terminal_state(3, 3);
        
        double q_before = sarsa.get_q_value(state, Action::RIGHT);
        
        // Update with terminal state (next_action doesn't matter for terminal states)
        sarsa.update(state, Action::RIGHT, 10.0, terminal_state, Action::UP, true);
        
        double q_after = sarsa.get_q_value(state, Action::RIGHT);
        
        // Should have updated with just the immediate reward (no future value)
        double expected_change = 0.5 * (10.0 - q_before); // α * (r + 0 - Q(s,a))
        double actual_change = q_after - q_before;
        
        assert_test(std::abs(actual_change - expected_change) < 1e-6, "Terminal state update");
    }
    
    void test_statistics() {
        if (verbose_) std::cout << "\n=== Testing Statistics ===" << std::endl;
        
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
        sarsa.set_actions(actions);
        
        const auto& stats = sarsa.get_statistics();
        assert_test(stats.total_updates == 0, "Initial update count");
        assert_test(stats.total_actions == 0, "Initial action count");
        
        GridState state(1, 1);
        GridState next_state(1, 0);
        
        // Perform some operations
        sarsa.select_action(state);
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        
        const auto& stats_after = sarsa.get_statistics();
        assert_test(stats_after.total_actions == 1, "Action count after selection");
        assert_test(stats_after.total_updates == 1, "Update count after update");
        assert_test(stats_after.cumulative_reward == -1.0, "Cumulative reward");
    }
    
    void test_reset() {
        if (verbose_) std::cout << "\n=== Testing Reset ===" << std::endl;
        
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
        sarsa.set_actions(actions);
        
        GridState state(1, 1);
        GridState next_state(1, 0);
        
        // Perform some operations
        sarsa.select_action(state);
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        
        // Verify state changed
        assert_test(sarsa.get_q_value(state, Action::UP) != 0.0, "Q-value set before reset");
        assert_test(sarsa.get_statistics().total_updates > 0, "Updates before reset");
        
        // Reset and verify
        sarsa.reset();
        assert_test(sarsa.get_q_value(state, Action::UP) == 0.0, "Q-value reset");
        assert_test(sarsa.get_statistics().total_updates == 0, "Statistics reset");
    }
    
    void test_simple_learning() {
        if (verbose_) std::cout << "\n=== Testing Simple Learning ===" << std::endl;
        
        // Simple 2x2 grid with goal at (1,1)
        GridWorld env(2, 2, GridState(1, 1));
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        
        SARSA<GridState, Action> sarsa(0.5, 0.9, 0.1, actions);
        sarsa.set_actions(actions);
        
        GridState start = env.get_start_state();
        
        // Run a few episodes
        double total_reward = 0.0;
        int episodes = 20;
        
        for (int episode = 0; episode < episodes; ++episode) {
            GridState current_state = start;
            Action current_action = sarsa.select_action(current_state);
            double episode_reward = 0.0;
            
            for (int step = 0; step < 10 && !env.is_terminal(current_state); ++step) {
                auto [next_state, reward] = env.step(current_state, current_action);
                Action next_action = env.is_terminal(next_state) ? Action::UP : sarsa.select_action(next_state);
                
                sarsa.update(current_state, current_action, reward, next_state, next_action, env.is_terminal(next_state));
                
                episode_reward += reward;
                current_state = next_state;
                current_action = next_action;
            }
            
            total_reward += episode_reward;
        }
        
        // Agent should have learned something (average reward should improve)
        double avg_reward = total_reward / episodes;
        assert_test(avg_reward > -10.0, "Learning progress"); // Arbitrary threshold
        
        // Q-values should be non-zero for visited states
        bool found_nonzero_q = false;
        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                GridState state(x, y);
                for (Action action : actions) {
                    if (sarsa.get_q_value(state, action) != 0.0) {
                        found_nonzero_q = true;
                        break;
                    }
                }
                if (found_nonzero_q) break;
            }
            if (found_nonzero_q) break;
        }
        assert_test(found_nonzero_q, "Non-zero Q-values after learning");
    }
    
    void run_all_tests() {
        std::cout << "Running SARSA Basic Functionality Tests..." << std::endl;
        
        test_constructor();
        test_parameter_setting();
        test_q_value_operations();
        test_action_selection();
        test_best_action();
        test_terminal_state_handling();
        test_statistics();
        test_reset();
        test_simple_learning();
        
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed_tests_ << "/" << total_tests_ << std::endl;
        
        if (passed_tests_ == total_tests_) {
            std::cout << "🎉 All tests passed!" << std::endl;
        } else {
            std::cout << "❌ Some tests failed!" << std::endl;
        }
    }
};

int main() {
    try {
        SARSABasicTest test(true); // verbose = true
        test.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
