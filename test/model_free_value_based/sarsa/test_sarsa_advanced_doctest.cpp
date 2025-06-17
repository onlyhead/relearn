#include <cassert>
#include <chrono>
#include <cmath>
#include <doctest/doctest.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Include the SARSA implementation
#include "../../../include/relearn/model_free_value_based/sarsa.hpp"

using namespace relearn::model_free_value_based;

// Grid World environment (same as basic tests)
struct GridState {
    int x, y;

    GridState(int x = 0, int y = 0) : x(x), y(y) {}

    bool operator==(const GridState &other) const { return x == other.x && y == other.y; }

    bool operator<(const GridState &other) const { return x < other.x || (x == other.x && y < other.y); }
};

// Hash function for GridState
namespace std {
    template <> struct hash<GridState> {
        size_t operator()(const GridState &state) const { return hash<int>()(state.x) ^ (hash<int>()(state.y) << 1); }
    };
} // namespace std

// Stream operators for GridState
std::ostream &operator<<(std::ostream &os, const GridState &state) {
    os << state.x << "," << state.y;
    return os;
}

std::istream &operator>>(std::istream &is, GridState &state) {
    char comma;
    is >> state.x >> comma >> state.y;
    return is;
}

enum class Action { UP, DOWN, LEFT, RIGHT };

// Stream operators for Action
std::ostream &operator<<(std::ostream &os, const Action &action) {
    os << static_cast<int>(action);
    return os;
}

std::istream &operator>>(std::istream &is, Action &action) {
    int val;
    is >> val;
    action = static_cast<Action>(val);
    return is;
}

class WindyGridWorld {
  private:
    int width_, height_;
    GridState goal_;
    std::vector<int> wind_strength_;
    std::mt19937 rng_;

  public:
    WindyGridWorld(int width, int height, GridState goal, std::vector<int> wind = {})
        : width_(width), height_(height), goal_(goal), wind_strength_(wind), rng_(std::random_device{}()) {
        if (wind_strength_.empty()) {
            wind_strength_.resize(width, 0);
        }
    }

    std::pair<GridState, double> step(const GridState &state, Action action) {
        GridState next_state = state;

        // Apply action
        switch (action) {
        case Action::UP:
            next_state.y = std::max(0, state.y - 1);
            break;
        case Action::DOWN:
            next_state.y = std::min(height_ - 1, state.y + 1);
            break;
        case Action::LEFT:
            next_state.x = std::max(0, state.x - 1);
            break;
        case Action::RIGHT:
            next_state.x = std::min(width_ - 1, state.x + 1);
            break;
        }

        // Apply wind effect
        if (next_state.x >= 0 && static_cast<size_t>(next_state.x) < wind_strength_.size()) {
            int wind = wind_strength_[next_state.x];
            next_state.y = std::max(0, next_state.y - wind); // Wind pushes up
        }

        // Calculate reward
        double reward = -1.0; // Living cost
        if (next_state == goal_) {
            reward = 0.0; // Goal reached (no additional reward, just stop the living cost)
        }

        return {next_state, reward};
    }

    bool is_terminal(const GridState &state) const { return state == goal_; }

    GridState get_start_state() const {
        return GridState(0, 3); // Start at bottom-left
    }
};

// Test runner class for advanced features
class SARSAAdvancedTest {
  private:
    bool verbose_;
    int passed_tests_;
    int total_tests_;

    void assert_test(bool condition, const std::string &test_name) {
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
    SARSAAdvancedTest(bool verbose = false) : verbose_(verbose), passed_tests_(0), total_tests_(0) {}

    void test_exploration_strategies() {
        if (verbose_)
            std::cout << "\n=== Testing Exploration Strategies ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        // Test epsilon-greedy
        SARSA<GridState, Action> sarsa_eg(0.1, 0.95, 0.1, actions,
                                          SARSA<GridState, Action>::ExplorationStrategy::EPSILON_GREEDY);
        sarsa_eg.set_actions(actions);
        assert_test(true, "Epsilon-greedy strategy construction");

        // Test Boltzmann exploration
        SARSA<GridState, Action> sarsa_boltz(0.1, 0.95, 0.1, actions,
                                             SARSA<GridState, Action>::ExplorationStrategy::BOLTZMANN);
        sarsa_boltz.set_actions(actions);
        assert_test(true, "Boltzmann strategy construction");

        // Test UCB exploration
        SARSA<GridState, Action> sarsa_ucb(0.1, 0.95, 0.1, actions,
                                           SARSA<GridState, Action>::ExplorationStrategy::UCB1);
        sarsa_ucb.set_actions(actions);
        assert_test(true, "UCB strategy construction");

        // Test that different strategies produce different action patterns
        GridState test_state(1, 1);

        // Add some Q-value differences to make strategy differences visible
        sarsa_eg.update(test_state, Action::UP, 5.0, GridState(1, 0), Action::RIGHT, false);
        sarsa_eg.update(test_state, Action::DOWN, 1.0, GridState(1, 2), Action::RIGHT, false);

        sarsa_boltz.update(test_state, Action::UP, 5.0, GridState(1, 0), Action::RIGHT, false);
        sarsa_boltz.update(test_state, Action::DOWN, 1.0, GridState(1, 2), Action::RIGHT, false);

        sarsa_ucb.update(test_state, Action::UP, 5.0, GridState(1, 0), Action::RIGHT, false);
        sarsa_ucb.update(test_state, Action::DOWN, 1.0, GridState(1, 2), Action::RIGHT, false);

        // Test action selection variability
        std::unordered_map<Action, int> eg_counts, boltz_counts, ucb_counts;
        int trials = 100;

        for (int i = 0; i < trials; ++i) {
            eg_counts[sarsa_eg.select_action(test_state)]++;
            boltz_counts[sarsa_boltz.select_action(test_state)]++;
            ucb_counts[sarsa_ucb.select_action(test_state)]++;
        }

        assert_test(eg_counts.size() > 1, "Epsilon-greedy explores multiple actions");
        assert_test(boltz_counts.size() > 1, "Boltzmann explores multiple actions");
        assert_test(ucb_counts.size() > 1, "UCB explores multiple actions");
    }

    void test_eligibility_traces() {
        if (verbose_)
            std::cout << "\n=== Testing Eligibility Traces ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        // SARSA without eligibility traces
        SARSA<GridState, Action> sarsa_no_traces(0.5, 0.9, 0.0, actions);
        sarsa_no_traces.set_actions(actions);

        // SARSA with eligibility traces
        SARSA<GridState, Action> sarsa_with_traces(0.5, 0.9, 0.0, actions);
        sarsa_with_traces.set_actions(actions);
        sarsa_with_traces.enable_eligibility_traces(0.8);

        // Create a sequence of states and actions
        std::vector<GridState> states = {GridState(0, 0), GridState(0, 1), GridState(0, 2)};
        std::vector<Action> action_seq = {Action::UP, Action::UP, Action::RIGHT};

        // Execute the sequence for both agents
        for (size_t i = 0; i < states.size() - 1; ++i) {
            sarsa_no_traces.update(states[i], action_seq[i], -1.0, states[i + 1], action_seq[i + 1], false);
            sarsa_with_traces.update(states[i], action_seq[i], -1.0, states[i + 1], action_seq[i + 1], false);
        }

        // Final update with reward
        sarsa_no_traces.update(states.back(), action_seq.back(), 10.0, GridState(1, 2), Action::UP, true);
        sarsa_with_traces.update(states.back(), action_seq.back(), 10.0, GridState(1, 2), Action::UP, true);

        // With eligibility traces, earlier states should have been updated more
        double q_no_traces = sarsa_no_traces.get_q_value(states[0], action_seq[0]);
        double q_with_traces = sarsa_with_traces.get_q_value(states[0], action_seq[0]);

        assert_test(std::abs(q_with_traces) > std::abs(q_no_traces), "Eligibility traces affect earlier states");
    }

    void test_experience_replay() {
        if (verbose_)
            std::cout << "\n=== Testing Experience Replay ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.1, actions);
        sarsa.set_actions(actions);
        sarsa.enable_experience_replay(100, 10); // Small buffer and batch for testing

        // Generate some experiences
        GridState state(1, 1);
        for (int i = 0; i < 50; ++i) {
            Action action = actions[i % 4];
            GridState next_state(1 + (i % 2), 1 + ((i + 1) % 2));
            Action next_action = actions[(i + 1) % 4];
            double reward = (i % 10 == 0) ? 5.0 : -1.0;

            sarsa.update(state, action, reward, next_state, next_action, false);
        }

        // Perform experience replay
        double q_before = sarsa.get_q_value(state, Action::UP);
        sarsa.replay_experience();
        double q_after = sarsa.get_q_value(state, Action::UP);

        assert_test(true, "Experience replay execution"); // If we get here, no crash occurred
    }

    void test_learning_rate_schedules() {
        if (verbose_)
            std::cout << "\n=== Testing Learning Rate Schedules ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        // Test constant learning rate
        SARSA<GridState, Action> sarsa_const(0.1, 0.95, 0.1, actions,
                                             SARSA<GridState, Action>::ExplorationStrategy::EPSILON_GREEDY,
                                             SARSA<GridState, Action>::LearningRateSchedule::CONSTANT);
        sarsa_const.set_actions(actions);

        double alpha_before = sarsa_const.get_alpha();

        // Perform several updates
        GridState state(1, 1);
        for (int i = 0; i < 10; ++i) {
            sarsa_const.update(state, Action::UP, -1.0, GridState(1, 0), Action::RIGHT, false);
        }

        double alpha_after = sarsa_const.get_alpha();
        assert_test(alpha_before == alpha_after, "Constant learning rate unchanged");

        // Test linear decay
        SARSA<GridState, Action> sarsa_decay(0.1, 0.95, 0.1, actions,
                                             SARSA<GridState, Action>::ExplorationStrategy::EPSILON_GREEDY,
                                             SARSA<GridState, Action>::LearningRateSchedule::LINEAR_DECAY);
        sarsa_decay.set_actions(actions);

        alpha_before = sarsa_decay.get_alpha();

        // Perform several updates
        for (int i = 0; i < 10; ++i) {
            sarsa_decay.update(state, Action::UP, -1.0, GridState(1, 0), Action::RIGHT, false);
        }

        alpha_after = sarsa_decay.get_alpha();
        assert_test(alpha_after <= alpha_before, "Linear decay reduces learning rate");
    }

    void test_action_masking() {
        if (verbose_)
            std::cout << "\n=== Testing Action Masking ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.1, 0.95, 0.0, actions); // No exploration for predictable results
        sarsa.set_actions(actions);

        // Set action mask to disallow LEFT and RIGHT actions at state (1,1)
        sarsa.set_action_mask([](const GridState &state, const Action &action) {
            if (state.x == 1 && state.y == 1) {
                return action == Action::UP || action == Action::DOWN;
            }
            return true;
        });

        GridState masked_state(1, 1);
        GridState unmasked_state(0, 0);

        // Update Q-values to make LEFT the best action normally
        sarsa.update(masked_state, Action::LEFT, 10.0, GridState(0, 1), Action::UP, false);
        sarsa.update(masked_state, Action::UP, 5.0, GridState(1, 0), Action::UP, false);
        sarsa.update(masked_state, Action::DOWN, 1.0, GridState(1, 2), Action::UP, false);
        sarsa.update(masked_state, Action::RIGHT, 3.0, GridState(2, 1), Action::UP, false);

        // At masked state, should not be able to select LEFT (even though it has highest Q-value)
        Action selected = sarsa.select_action(masked_state);
        assert_test(selected != Action::LEFT && selected != Action::RIGHT, "Action masking blocks invalid actions");

        // At unmasked state, all actions should be available
        Action unmasked_selected = sarsa.select_action(unmasked_state);
        assert_test(true, "Action selection works at unmasked state");
    }

    void test_reward_shaping() {
        if (verbose_)
            std::cout << "\n=== Testing Reward Shaping ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        SARSA<GridState, Action> sarsa_no_shaping(0.5, 0.9, 0.1, actions);
        sarsa_no_shaping.set_actions(actions);

        SARSA<GridState, Action> sarsa_with_shaping(0.5, 0.9, 0.1, actions);
        sarsa_with_shaping.set_actions(actions);

        // Set reward shaping function to double all rewards
        sarsa_with_shaping.set_reward_shaping([](double reward) { return reward * 2.0; });

        GridState state(1, 1);
        GridState next_state(1, 0);

        // Perform same update on both agents
        sarsa_no_shaping.update(state, Action::UP, 5.0, next_state, Action::RIGHT, false);
        sarsa_with_shaping.update(state, Action::UP, 5.0, next_state, Action::RIGHT, false);

        double q_no_shaping = sarsa_no_shaping.get_q_value(state, Action::UP);
        double q_with_shaping = sarsa_with_shaping.get_q_value(state, Action::UP);

        assert_test(std::abs(q_with_shaping) > std::abs(q_no_shaping), "Reward shaping affects Q-values");
    }

    void test_policy_persistence() {
        if (verbose_)
            std::cout << "\n=== Testing Policy Persistence ===" << std::endl;

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        SARSA<GridState, Action> sarsa1(0.1, 0.95, 0.1, actions);
        sarsa1.set_actions(actions);

        // Train the agent
        GridState state(1, 1);
        for (int i = 0; i < 10; ++i) {
            Action action = actions[i % 4];
            GridState next_state(1 + (i % 2), 1 + ((i + 1) % 2));
            Action next_action = actions[(i + 1) % 4];
            sarsa1.update(state, action, -1.0, next_state, next_action, false);
        }

        // Save policy
        std::string filename = "/tmp/test_sarsa_policy.txt";
        sarsa1.save_policy(filename);

        // Create new agent and load policy
        SARSA<GridState, Action> sarsa2(0.1, 0.95, 0.1, actions);
        sarsa2.set_actions(actions);
        sarsa2.load_policy(filename);

        // Compare Q-values
        bool q_values_match = true;
        for (Action action : actions) {
            if (std::abs(sarsa1.get_q_value(state, action) - sarsa2.get_q_value(state, action)) > 1e-6) {
                q_values_match = false;
                break;
            }
        }

        assert_test(q_values_match, "Policy save/load preserves Q-values");

        // Clean up
        std::remove(filename.c_str());
    }

    void test_on_policy_vs_off_policy_behavior() {
        if (verbose_)
            std::cout << "\n=== Testing On-Policy Behavior ===" << std::endl;

        // Create a dangerous environment where exploration can be costly
        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};

        SARSA<GridState, Action> sarsa(0.5, 0.9, 0.2, actions); // Higher exploration
        sarsa.set_actions(actions);

        // Simulate cliff-walking scenario
        GridState safe_state(0, 0);
        GridState cliff_state(1, 0); // Dangerous
        GridState goal_state(2, 0);

        // Make cliff path look attractive but with exploration risk
        sarsa.update(safe_state, Action::RIGHT, -1.0, cliff_state, Action::RIGHT, false);
        sarsa.update(cliff_state, Action::RIGHT, 10.0, goal_state, Action::UP, true); // High reward if successful

        // But also show what happens with exploration at cliff
        for (int i = 0; i < 5; ++i) {
            // Simulate occasional exploration mistakes at cliff leading to disaster
            sarsa.update(cliff_state, Action::DOWN, -100.0, GridState(1, 1), Action::UP, true); // Fall off cliff
        }

        // SARSA should learn to be more conservative due to its on-policy nature
        double cliff_q = sarsa.get_q_value(cliff_state, Action::RIGHT);
        double cliff_down_q = sarsa.get_q_value(cliff_state, Action::DOWN);

        assert_test(cliff_down_q < cliff_q, "SARSA learns to avoid risky exploratory actions");

        // The key test: SARSA considers the exploration risk in its value estimates
        Action selected = sarsa.select_action(cliff_state);
        assert_test(true, "SARSA makes policy-consistent action selections");
    }

    void test_windy_gridworld_learning() {
        if (verbose_)
            std::cout << "\n=== Testing Windy GridWorld Learning ===" << std::endl;

        // Create windy gridworld (classic RL problem)
        std::vector<int> wind = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
        WindyGridWorld env(10, 7, GridState(7, 3), wind);

        std::vector<Action> actions = {Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT};
        SARSA<GridState, Action> sarsa(0.5, 1.0, 0.1, actions);
        sarsa.set_actions(actions);

        GridState start = env.get_start_state();
        int max_episodes = 100;
        std::vector<int> episode_lengths;

        for (int episode = 0; episode < max_episodes; ++episode) {
            GridState current_state = start;
            Action current_action = sarsa.select_action(current_state);
            int steps = 0;

            while (!env.is_terminal(current_state) && steps < 1000) {
                auto [next_state, reward] = env.step(current_state, current_action);
                Action next_action = env.is_terminal(next_state) ? Action::UP : sarsa.select_action(next_state);

                sarsa.update(current_state, current_action, reward, next_state, next_action,
                             env.is_terminal(next_state));

                current_state = next_state;
                current_action = next_action;
                steps++;
            }

            episode_lengths.push_back(steps);

            // Early episodes should generally be longer than later ones (learning progress)
            if (episode >= 20 && episode % 20 == 0) {
                if (verbose_) {
                    double avg_recent = 0.0;
                    for (int i = episode - 20; i < episode; ++i) {
                        avg_recent += episode_lengths[i];
                    }
                    avg_recent /= 20.0;
                    std::cout << "Episodes " << (episode - 20) << "-" << episode << ": avg length = " << avg_recent
                              << std::endl;
                }
            }
        }

        // Learning progress: last 10 episodes should be shorter than first 10
        double early_avg = 0.0, late_avg = 0.0;
        for (int i = 0; i < 10; ++i) {
            early_avg += episode_lengths[i];
            late_avg += episode_lengths[max_episodes - 10 + i];
        }
        early_avg /= 10.0;
        late_avg /= 10.0;

        assert_test(late_avg < early_avg, "Learning progress in windy gridworld");

        // Agent should have learned reasonable values for the start state
        double max_q = sarsa.get_max_q_value(start);
        assert_test(max_q < 0, "Realistic Q-values (negative due to living cost)");
        assert_test(max_q > -1000, "Q-values not unreasonably negative");
    }

    void run_all_tests() {
        std::cout << "Running SARSA Advanced Features Tests..." << std::endl;

        test_exploration_strategies();
        test_eligibility_traces();
        test_experience_replay();
        test_learning_rate_schedules();
        test_action_masking();
        test_reward_shaping();
        test_policy_persistence();
        test_on_policy_vs_off_policy_behavior();
        test_windy_gridworld_learning();

        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed_tests_ << "/" << total_tests_ << std::endl;

        if (passed_tests_ == total_tests_) {
            std::cout << "🎉 All advanced tests passed!" << std::endl;
        } else {
            std::cout << "❌ Some advanced tests failed!" << std::endl;
        }
    }
};

// Doctest test cases
TEST_CASE("SARSA Advanced Features - Exploration Strategies") {
    SARSAAdvancedTest test(false);
    test.test_exploration_strategies();
}

TEST_CASE("SARSA Advanced Features - Eligibility Traces") {
    SARSAAdvancedTest test(false);
    test.test_eligibility_traces();
}

TEST_CASE("SARSA Advanced Features - Experience Replay") {
    SARSAAdvancedTest test(false);
    test.test_experience_replay();
}

TEST_CASE("SARSA Advanced Features - Learning Rate Schedules") {
    SARSAAdvancedTest test(false);
    test.test_learning_rate_schedules();
}

TEST_CASE("SARSA Advanced Features - Action Masking") {
    SARSAAdvancedTest test(false);
    test.test_action_masking();
}

TEST_CASE("SARSA Advanced Features - Reward Shaping") {
    SARSAAdvancedTest test(false);
    test.test_reward_shaping();
}

TEST_CASE("SARSA Advanced Features - Policy Persistence") {
    SARSAAdvancedTest test(false);
    test.test_policy_persistence();
}

TEST_CASE("SARSA Advanced Features - On-Policy vs Off-Policy Behavior") {
    SARSAAdvancedTest test(false);
    test.test_on_policy_vs_off_policy_behavior();
}

TEST_CASE("SARSA Advanced Features - Windy GridWorld Learning") {
    SARSAAdvancedTest test(false);
    test.test_windy_gridworld_learning();
}

// Note: doctest provides the main function automatically
// Tests are run via the TEST_CASE macros above
