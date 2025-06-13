#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("Hierarchical and Meta-RL algorithms basic functionality") {
    SUBCASE("Options Construction") {
        hierarchical_meta::Options<std::vector<double>, std::vector<double>> options;
        CHECK(true); // Placeholder
    }

    SUBCASE("Feudal Networks Construction") {
        hierarchical_meta::FeudalNetworks<std::vector<double>, std::vector<double>> feudal;
        CHECK(true); // Placeholder
    }

    SUBCASE("MAML Construction") {
        hierarchical_meta::MAML<std::vector<double>, std::vector<double>> maml;
        CHECK(true); // Placeholder
    }

    SUBCASE("RL^2 Construction") {
        hierarchical_meta::RLSquared<std::vector<double>, std::vector<double>> rl2;
        CHECK(true); // Placeholder
    }
}
