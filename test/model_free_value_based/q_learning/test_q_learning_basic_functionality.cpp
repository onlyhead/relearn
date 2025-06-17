#include <doctest/doctest.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <relearn/relearn.hpp>
#include <thread>
#include <vector>

using namespace relearn::model_free_value_based;

TEST_CASE("Q-Learning Basic Functionality - Constructor Validation") {
    std::vector<int> actions = {0, 1, 2, 3};

    SUBCASE("Valid constructor should not throw") {
        CHECK_NOTHROW(QLearning<int, int> agent(0.1, 0.9, 0.1, actions));
    }

    SUBCASE("Constructor creates agent with correct parameters") {
        QLearning<int, int> agent(0.1, 0.9, 0.1, actions);
        CHECK(agent.get_learning_rate() == doctest::Approx(0.1));
        CHECK(agent.get_discount_factor() == doctest::Approx(0.9));
        CHECK(agent.get_epsilon() == doctest::Approx(0.1));
    }
}

TEST_CASE("Q-Learning Basic Functionality - Basic Learning Update") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    int state = 0;
    int action = 1;
    double reward = 1.0;
    int next_state = 1;
    bool terminal = false;

    SUBCASE("Initial Q-value should be zero") {
        double initial_q = agent.get_q_value(state, action);
        CHECK(initial_q == doctest::Approx(0.0));
    }

    SUBCASE("Q-value increases after positive reward update") {
        double initial_q = agent.get_q_value(state, action);
        agent.update(state, action, reward, next_state, terminal);
        double updated_q = agent.get_q_value(state, action);
        CHECK(updated_q > initial_q);
    }
}

TEST_CASE("Q-Learning Basic Functionality - Action Selection") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.0, actions); // No exploration for deterministic testing

    int state = 0;
    
    SUBCASE("Selected action should be from available actions") {
        int selected_action = agent.select_action(state);
        bool valid_action = std::find(actions.begin(), actions.end(), selected_action) != actions.end();
        CHECK(valid_action);
    }

    SUBCASE("Should select action with highest Q-value when no exploration") {
        // Set one action to have higher Q-value
        agent.update(state, 1, 1.0, state + 1, false);
        int selected_action = agent.select_action(state);
        CHECK(selected_action == 1);
    }
}

TEST_CASE("Q-Learning Basic Functionality - Parameter Getters and Setters") {
    std::vector<int> actions = {0, 1, 2, 3};
    double learning_rate = 0.1;
    double discount_factor = 0.9;
    double exploration_rate = 0.1;
    
    QLearning<int, int> agent(learning_rate, discount_factor, exploration_rate, actions);

    SUBCASE("Parameter getters return correct values") {
        CHECK(agent.get_learning_rate() == doctest::Approx(learning_rate));
        CHECK(agent.get_discount_factor() == doctest::Approx(discount_factor));
        CHECK(agent.get_epsilon() == doctest::Approx(exploration_rate));
    }

    SUBCASE("Parameter setters work correctly") {
        agent.set_learning_rate(0.2);
        CHECK(agent.get_learning_rate() == doctest::Approx(0.2));
        
        agent.set_discount_factor(0.95);
        CHECK(agent.get_discount_factor() == doctest::Approx(0.95));
        
        agent.set_epsilon(0.2);
        CHECK(agent.get_epsilon() == doctest::Approx(0.2));
    }
}

TEST_CASE("Q-Learning Basic Functionality - Q-Value Operations") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    int state = 0;

    SUBCASE("Initial max Q-value should be zero") {
        double initial_max_q = agent.get_max_q_value(state);
        CHECK(initial_max_q == doctest::Approx(0.0));
    }

    SUBCASE("Best action should be from available actions") {
        int best_action = agent.get_best_action(state);
        bool valid_best_action = std::find(actions.begin(), actions.end(), best_action) != actions.end();
        CHECK(valid_best_action);
    }

    SUBCASE("Best action should be the highest Q-value action") {
        // Set action 2 to have highest Q-value
        agent.update(state, 2, 1.0, state + 1, false);
        int best_action = agent.get_best_action(state);
        CHECK(best_action == 2);
    }
}

TEST_CASE("Q-Learning Basic Functionality - Advanced Features") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    SUBCASE("Double Q-learning can be enabled") {
        CHECK_NOTHROW(agent.set_double_q_learning(true));
    }

    SUBCASE("Eligibility traces can be enabled") {
        CHECK_NOTHROW(agent.set_eligibility_traces(true, 0.9));
    }

    SUBCASE("Experience replay can be enabled") {
        CHECK_NOTHROW(agent.set_experience_replay(true, 1000, 32));
    }

    SUBCASE("Action masking can be set") {
        std::function<bool(const int&, const int&)> mask_func = [](const int& state, const int& action) { return true; };
        CHECK_NOTHROW(agent.set_action_mask(mask_func));
    }

    SUBCASE("Reward shaping can be set") {
        std::function<double(double)> shaping_func = [](double reward) { return reward; };
        CHECK_NOTHROW(agent.set_reward_shaping(shaping_func));
    }
}

TEST_CASE("Q-Learning Basic Functionality - Statistics") {
    std::vector<int> actions = {0, 1, 2, 3};
    QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

    SUBCASE("Initial statistics should be zero") {
        auto stats = agent.get_statistics();
        CHECK(stats.total_updates == 0);
        CHECK(stats.total_actions == 0);
        CHECK(stats.cumulative_reward == doctest::Approx(0.0));
        CHECK(stats.average_q_value == doctest::Approx(0.0));
        CHECK(stats.exploration_ratio == doctest::Approx(0.0));
    }

    SUBCASE("Statistics update after learning") {
        agent.update(0, 1, 1.0, 1, false);
        auto stats = agent.get_statistics();
        CHECK(stats.total_updates == 1);
    }
}
