#pragma once

#include <memory>
#include <thread>
#include <vector>

namespace relearn {
    namespace model_free_actor_critic {

        /**
         * @brief Advantage Actor-Critic (A2C) implementation
         *
         * Synchronous version of A3C with multiple parallel workers.
         * Combines policy gradient (actor) with value estimation (critic).
         */
        template <typename StateType, typename ActionType> class A2C {
          private:
            // Actor and critic networks
            int num_workers_ = 4;

          public:
            inline A2C() = default;

            // Method signatures to be implemented
            inline void collect_experiences();
            inline void update_networks();
            inline ActionType sample_action(const StateType &state);
            inline double compute_advantage(double reward, double value, double next_value);
            inline void sync_workers();
        };

        /**
         * @brief Asynchronous Advantage Actor-Critic (A3C) implementation
         *
         * Asynchronous version with independent worker threads.
         * Each worker updates global networks asynchronously.
         */
        template <typename StateType, typename ActionType> class A3C {
          private:
            // Global networks and worker threads
            std::vector<std::thread> workers_;

          public:
            inline A3C() = default;

            // Method signatures to be implemented
            inline void start_training();
            inline void worker_thread(int worker_id);
            inline void update_global_networks();
            inline void stop_training();
        };

    } // namespace model_free_actor_critic
} // namespace relearn
