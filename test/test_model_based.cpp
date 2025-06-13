#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("Model-based algorithms basic functionality") {
    SUBCASE("PILCO Construction") {
        model_based::PILCO<std::vector<double>, std::vector<double>> pilco;
        CHECK(true); // Placeholder
    }

    SUBCASE("MBPO Construction") {
        model_based::MBPO<std::vector<double>, std::vector<double>> mbpo;
        CHECK(true); // Placeholder
    }

    SUBCASE("Dreamer Construction") {
        model_based::Dreamer<std::vector<double>, std::vector<double>> dreamer;
        CHECK(true); // Placeholder
    }

    SUBCASE("PETS Construction") {
        model_based::PETS<std::vector<double>, std::vector<double>> pets;
        CHECK(true); // Placeholder
    }
}
