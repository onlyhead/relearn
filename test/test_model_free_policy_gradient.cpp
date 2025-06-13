#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("REINFORCE basic functionality") {
    model_free_policy_gradient::REINFORCE<std::vector<double>, std::vector<double>> reinforce;

    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}

TEST_CASE("TRPO basic functionality") {
    model_free_policy_gradient::TRPO<std::vector<double>, std::vector<double>> trpo;

    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}

TEST_CASE("PPO basic functionality") {
    model_free_policy_gradient::PPO<std::vector<double>, std::vector<double>> ppo;

    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}
