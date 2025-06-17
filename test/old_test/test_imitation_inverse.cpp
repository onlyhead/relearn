#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("Imitation Learning algorithms basic functionality") {
    SUBCASE("Behavioral Cloning Construction") {
        imitation_inverse::BehavioralCloning<std::vector<double>, std::vector<double>> bc;
        CHECK(true); // Placeholder
    }

    SUBCASE("DAgger Construction") {
        auto expert_policy = [](const std::vector<double> &state) -> std::vector<double> {
            return std::vector<double>(1, 0.0); // Dummy expert policy
        };
        imitation_inverse::DAgger<std::vector<double>, std::vector<double>> dagger(expert_policy);
        CHECK(true); // Placeholder
    }

    SUBCASE("GAIL Construction") {
        imitation_inverse::GAIL<std::vector<double>, std::vector<double>> gail;
        CHECK(true); // Placeholder
    }
}
