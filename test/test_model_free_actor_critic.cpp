#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("A2C/A3C basic functionality") {
    model_free_actor_critic::A2C<std::vector<double>, std::vector<double>> a2c;
    
    SUBCASE("A2C Construction") {
        CHECK(true); // Placeholder
    }
    
    SUBCASE("A3C Construction") {
        model_free_actor_critic::A3C<std::vector<double>, std::vector<double>> a3c;
        CHECK(true); // Placeholder
    }
}

TEST_CASE("DDPG basic functionality") {
    model_free_actor_critic::DDPG<std::vector<double>, std::vector<double>> ddpg;
    
    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}

TEST_CASE("TD3 basic functionality") {
    model_free_actor_critic::TD3<std::vector<double>, std::vector<double>> td3;
    
    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}

TEST_CASE("SAC basic functionality") {
    model_free_actor_critic::SAC<std::vector<double>, std::vector<double>> sac;
    
    SUBCASE("Construction") {
        CHECK(true); // Placeholder
    }
}
