#include <iostream>
#include <relearn/relearn.hpp>
#include <vector>

using namespace relearn;

// Simple grid world environment for demonstration
class SimpleGridWorld : public common::Environment<std::vector<int>, int> {
  private:
    int grid_size_ = 5;
    std::vector<int> current_state_;
    std::vector<int> goal_state_;

  public:
    SimpleGridWorld() : current_state_{0, 0}, goal_state_{4, 4} {}

    std::vector<int> reset() override {
        current_state_ = {0, 0};
        return current_state_;
    }

    std::tuple<std::vector<int>, double, bool> step(const int &action) override {
        // Actions: 0=up, 1=right, 2=down, 3=left
        auto next_state = current_state_;

        switch (action) {
        case 0:
            if (current_state_[0] > 0)
                next_state[0]--;
            break; // up
        case 1:
            if (current_state_[1] < grid_size_ - 1)
                next_state[1]++;
            break; // right
        case 2:
            if (current_state_[0] < grid_size_ - 1)
                next_state[0]++;
            break; // down
        case 3:
            if (current_state_[1] > 0)
                next_state[1]--;
            break; // left
        }

        current_state_ = next_state;

        // Reward: +1 for reaching goal, -0.01 for each step
        double reward = (current_state_ == goal_state_) ? 1.0 : -0.01;
        bool done = (current_state_ == goal_state_);

        return std::make_tuple(current_state_, reward, done);
    }

    std::vector<int> get_action_space() const override { return {0, 1, 2, 3}; }

    bool is_terminal(const std::vector<int> &state) const override { return state == goal_state_; }
};

int main() {
    std::cout << "ReLearn Library v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
    std::cout << "Reinforcement Learning Library Structure Demo" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Create environment
    SimpleGridWorld env;

    std::cout << "\n1. Model-Free, Value-Based Algorithms:" << std::endl;
    {
        model_free_value_based::QLearning<int, int> q_learning;
        std::cout << "   - Q-Learning: Created successfully" << std::endl;
        (void)q_learning;

        model_free_value_based::DQN<std::vector<int>, int> dqn;
        std::cout << "   - DQN: Created successfully" << std::endl;
        (void)dqn;

        model_free_value_based::DoubleDQN<std::vector<int>, int> double_dqn;
        std::cout << "   - Double DQN: Created successfully" << std::endl;
        (void)double_dqn;
    }

    std::cout << "\n2. Model-Free, Policy-Gradient Algorithms:" << std::endl;
    {
        model_free_policy_gradient::REINFORCE<std::vector<int>, int> reinforce;
        std::cout << "   - REINFORCE: Created successfully" << std::endl;
        (void)reinforce; // Suppress unused warning

        model_free_policy_gradient::TRPO<std::vector<int>, int> trpo;
        std::cout << "   - TRPO: Created successfully" << std::endl;
        (void)trpo; // Suppress unused warning

        model_free_policy_gradient::PPO<std::vector<int>, int> ppo;
        std::cout << "   - PPO: Created successfully" << std::endl;
        (void)ppo; // Suppress unused warning
    }

    std::cout << "\n3. Model-Free, Actor-Critic Algorithms:" << std::endl;
    {
        model_free_actor_critic::A2C<std::vector<int>, int> a2c;
        std::cout << "   - A2C: Created successfully" << std::endl;
        (void)a2c;

        model_free_actor_critic::DDPG<std::vector<int>, int> ddpg;
        std::cout << "   - DDPG: Created successfully" << std::endl;
        (void)ddpg;

        model_free_actor_critic::TD3<std::vector<int>, int> td3;
        std::cout << "   - TD3: Created successfully" << std::endl;
        (void)td3;

        model_free_actor_critic::SAC<std::vector<int>, int> sac;
        std::cout << "   - SAC: Created successfully" << std::endl;
        (void)sac;
    }

    std::cout << "\n4. Model-Based RL Algorithms:" << std::endl;
    {
        model_based::PILCO<std::vector<int>, int> pilco;
        std::cout << "   - PILCO: Created successfully" << std::endl;
        (void)pilco;

        model_based::MBPO<std::vector<int>, int> mbpo;
        std::cout << "   - MBPO: Created successfully" << std::endl;
        (void)mbpo;

        model_based::Dreamer<std::vector<int>, int> dreamer;
        std::cout << "   - Dreamer: Created successfully" << std::endl;
        (void)dreamer;
    }

    std::cout << "\n5. Imitation & Inverse RL Algorithms:" << std::endl;
    {
        imitation_inverse::BehavioralCloning<std::vector<int>, int> bc;
        std::cout << "   - Behavioral Cloning: Created successfully" << std::endl;
        (void)bc;

        imitation_inverse::GAIL<std::vector<int>, int> gail;
        std::cout << "   - GAIL: Created successfully" << std::endl;
        (void)gail;
    }

    std::cout << "\n6. Common Utilities:" << std::endl;
    {
        common::ReplayBuffer<std::vector<int>, int> buffer(1000);
        std::cout << "   - Replay Buffer: Created successfully (capacity: 1000)" << std::endl;

        // Test some utility functions
        std::vector<double> test_rewards = {1.0, 0.5, 0.0, 1.0};
        auto returns = common::Utils::compute_returns(test_rewards);
        std::cout << "   - Utility functions: Working (computed " << returns.size() << " returns)" << std::endl;
    }

    std::cout << "\n7. Evolutionary & Black-Box Methods:" << std::endl;
    {
        evolutionary_blackbox::CMAES<int> cmaes(10);
        std::cout << "   - CMA-ES: Created successfully (10D parameter space)" << std::endl;

        evolutionary_blackbox::NES<int> nes(10, 50);
        std::cout << "   - NES: Created successfully (10D, population 50)" << std::endl;
    }

    std::cout << "\n8. Q-Learning Demo:" << std::endl;
    {
        // Simple 2x2 grid world demo
        std::vector<int> actions = {0, 1, 2, 3}; // up, down, left, right
        model_free_value_based::QLearning<int, int> agent(0.1, 0.9, 0.1, actions);

        std::cout << "   - Training agent on simple grid world..." << std::endl;

        // Train for a few episodes
        for (int episode = 0; episode < 10; ++episode) {
            // Simple episode: state 0 -> 1 -> 3 (goal with reward 10)
            agent.update(0, 3, 0.0, 1, false); // Move right: 0 -> 1
            agent.update(1, 1, 10.0, 3, true); // Move down: 1 -> 3 (goal!)
        }

        std::cout << "   - Q(0,3) = " << agent.get_q_value(0, 3) << std::endl;
        std::cout << "   - Q(1,1) = " << agent.get_q_value(1, 1) << std::endl;
        std::cout << "   - Best action from state 1: " << agent.get_best_action(1) << std::endl;
        std::cout << "   - Q-table size: " << agent.get_q_table_size() << " entries" << std::endl;
    }

    std::cout << "\nLibrary structure created successfully!" << std::endl;
    std::cout << "All algorithm classes are available and ready for implementation." << std::endl;

    return 0;
}
