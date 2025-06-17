#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("Evolutionary and Black-box algorithms basic functionality") {
    SUBCASE("CMA-ES Construction") {
        evolutionary_blackbox::CMAES<int> cmaes(10); // 10-dimensional parameter space
        CHECK(true);                                 // Placeholder
    }

    SUBCASE("NES Construction") {
        evolutionary_blackbox::NES<int> nes(10, 50); // 10-dim parameters, 50 population size
        CHECK(true);                                 // Placeholder
    }
}
