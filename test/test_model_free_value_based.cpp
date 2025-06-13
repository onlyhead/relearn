#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("Q-Learning basic functionality") {
    model_free_value_based::QLearning<int, int> q_learning;

    SUBCASE("Construction") {
        // Test that Q-Learning object can be constructed
        CHECK(true); // Placeholder - will be implemented when methods are added
    }

    SUBCASE("Method signatures exist") {
        // Test that required methods exist (will compile if signatures are correct)
        // These will be implemented when the actual algorithm logic is added
        CHECK(true);
    }
}

TEST_CASE("DQN basic functionality") {
    model_free_value_based::DQN<std::vector<double>, int> dqn;

    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}

TEST_CASE("Common utilities") {
    SUBCASE("ReplayBuffer construction") {
        common::ReplayBuffer<int, int> buffer(1000);
        CHECK(buffer.size() == 0);
    }
}
