#include <doctest/doctest.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
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

TEST_CASE("SARSA Constructor") {
    SUBCASE("Default parameters") {
        SARSA<GridState, Action> sarsa;
        CHECK(sarsa.get_alpha() == doctest::Approx(0.1));
        CHECK(sarsa.get_gamma() == doctest::Approx(0.95));
        CHECK(sarsa.get_epsilon() == doctest::Approx(0.1));
    }
    
    SUBCASE("Custom parameters") {
        SARSA<GridState, Action> sarsa(0.2, 0.9, 0.05);
        CHECK(sarsa.get_alpha() == doctest::Approx(0.2));
        CHECK(sarsa.get_gamma() == doctest::Approx(0.9));
        CHECK(sarsa.get_epsilon() == doctest::Approx(0.05));
    }
    
    SUBCASE("Constructor with actions") {
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
        CHECK(sarsa.get_alpha() == doctest::Approx(0.1));
        CHECK(sarsa.get_gamma() == doctest::Approx(0.95));
        CHECK(sarsa.get_epsilon() == doctest::Approx(0.1));
    }
}

TEST_CASE("SARSA Parameter Setting") {
    SARSA<GridState, Action> sarsa;
    
    SUBCASE("Alpha setting") {
        sarsa.set_alpha(0.5);
        CHECK(sarsa.get_alpha() == doctest::Approx(0.5));
    }
    
    SUBCASE("Gamma setting") {
        sarsa.set_gamma(0.8);
        CHECK(sarsa.get_gamma() == doctest::Approx(0.8));
    }
    
    SUBCASE("Epsilon setting") {
        sarsa.set_epsilon(0.2);
        CHECK(sarsa.get_epsilon() == doctest::Approx(0.2));
    }
    
    SUBCASE("Lambda setting") {
        sarsa.set_lambda(0.7);
        CHECK(sarsa.get_lambda() == doctest::Approx(0.7));
    }
    
    SUBCASE("Boundary constraints") {
        sarsa.set_gamma(1.5); // Should be clamped to 1.0
        CHECK(sarsa.get_gamma() == doctest::Approx(1.0));
        
        sarsa.set_gamma(-0.1); // Should be clamped to 0.0
        CHECK(sarsa.get_gamma() == doctest::Approx(0.0));
    }
}

TEST_CASE("SARSA Q-Value Operations") {
    std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
    SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
    sarsa.set_actions(actions);
    
    GridState state(1, 1);
    
    SUBCASE("Initial Q-values") {
        CHECK(sarsa.get_q_value(state, Action::UP) == doctest::Approx(0.0));
    }
    
    SUBCASE("Basic update") {
        GridState next_state(1, 0);
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        
        // Q-value should have changed
        double q_after_update = sarsa.get_q_value(state, Action::UP);
        CHECK(q_after_update != doctest::Approx(0.0));
    }
    
    SUBCASE("Multiple updates") {
        GridState next_state(1, 0);
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        double q_before = sarsa.get_q_value(state, Action::UP);
        
        sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
        double q_after = sarsa.get_q_value(state, Action::UP);
        
        CHECK(q_after != doctest::Approx(q_before));
    }
}

TEST_CASE("SARSA Action Selection") {
    std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
    
    SUBCASE("Consistent action selection with zero exploration") {
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.0, actions); // No exploration
        sarsa.set_actions(actions);
        
        GridState state(1, 1);
        
        // With zero exploration, should always select same action initially
        Action first_action = sarsa.select_action(state);
        Action second_action = sarsa.select_action(state);
        CHECK(first_action == second_action);
    }
    
    SUBCASE("Different actions with full exploration") {
        SARSA<GridState, Action> sarsa(0.1, 0.95, 1.0, actions); // Always explore
        sarsa.set_actions(actions);
        
        GridState state(1, 1);
        
        bool different_actions = false;
        Action base_action = sarsa.select_action(state);
        
        for (int i = 0; i < 20; ++i) {
            if (sarsa.select_action(state) != base_action) {
                different_actions = true;
                break;
            }
        }
        CHECK(different_actions);
    }
}

TEST_CASE("SARSA Best Action") {
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
    CHECK(best == Action::UP);
}

TEST_CASE("SARSA Terminal State Handling") {
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
    
    CHECK(std::abs(actual_change - expected_change) < 1e-6);
}

TEST_CASE("SARSA Statistics") {
    std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
    SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
    sarsa.set_actions(actions);
    
    const auto& stats = sarsa.get_statistics();
    CHECK(stats.total_updates == 0);
    CHECK(stats.total_actions == 0);
    
    GridState state(1, 1);
    GridState next_state(1, 0);
    
    // Perform some operations
    sarsa.select_action(state);
    sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
    
    const auto& stats_after = sarsa.get_statistics();
    CHECK(stats_after.total_actions == 1);
    CHECK(stats_after.total_updates == 1);
    CHECK(stats_after.cumulative_reward == doctest::Approx(-1.0));
}

TEST_CASE("SARSA Reset") {
    std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
    SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
    sarsa.set_actions(actions);
    
    GridState state(1, 1);
    GridState next_state(1, 0);
    
    // Perform some operations
    sarsa.select_action(state);
    sarsa.update(state, Action::UP, -1.0, next_state, Action::RIGHT, false);
    
    // Verify state changed
    CHECK(sarsa.get_q_value(state, Action::UP) != doctest::Approx(0.0));
    CHECK(sarsa.get_statistics().total_updates > 0);
    
    // Reset and verify
    sarsa.reset();
    CHECK(sarsa.get_q_value(state, Action::UP) == doctest::Approx(0.0));
    CHECK(sarsa.get_statistics().total_updates == 0);
}

TEST_CASE("SARSA Simple Learning") {
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
    CHECK(avg_reward > -10.0); // Arbitrary threshold
    
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
    CHECK(found_nonzero_q);
}
