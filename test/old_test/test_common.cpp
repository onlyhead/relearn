#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <relearn/relearn.hpp>

using namespace relearn;

TEST_CASE("Common utilities functionality") {
    SUBCASE("ReplayBuffer basic operations") {
        common::ReplayBuffer<int, int> buffer(100);

        CHECK(buffer.size() == 0);
        CHECK_FALSE(buffer.can_sample(10));

        // Add some transitions
        buffer.add(1, 2, 0.5, 3, false);
        buffer.add(3, 1, 1.0, 5, true);

        CHECK(buffer.size() == 2);
        CHECK(buffer.can_sample(2));
        CHECK_FALSE(buffer.can_sample(5));
    }

    SUBCASE("Utils statistical functions") {
        std::vector<double> rewards = {1.0, 0.5, 0.0, 1.0};
        std::vector<double> values = {0.8, 0.4, 0.1, 0.9};

        // Test that functions exist and can be called
        auto returns = common::Utils::compute_returns(rewards);
        CHECK(returns.size() == rewards.size());

        auto gae = common::Utils::compute_gae(rewards, values);
        CHECK(gae >= 0.0); // GAE should be computed

        std::vector<double> test_values = {1.0, 2.0, 3.0};
        auto normalized = common::Utils::normalize(test_values);
        CHECK(normalized.size() == test_values.size());

        std::vector<double> logits = {1.0, 2.0, 3.0};
        auto softmax_result = common::Utils::softmax(logits);
        CHECK(softmax_result.size() == logits.size());

        double log_sum_exp_result = common::Utils::log_sum_exp(logits);
        CHECK(log_sum_exp_result > 0.0);
    }
}
