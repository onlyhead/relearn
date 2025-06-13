Here’s a rundown of the main families of reinforcement-learning algorithms, with a few representative methods in each and notes on their suitability for robotics (continuous control, sample efficiency, etc.):

---

## 1. Model-Free, Value-Based

These learn a value function $Q(s,a)$ and derive a policy by acting greedily (or ε-greedy) w\.r.t. it.

* **Q-Learning**

  * Tabular; discrete states/actions.
  * Rarely used directly in robotics unless you discretize heavily.
* **Deep Q-Network (DQN)**

  * Uses a neural network to approximate $Q$.
  * Works for discrete action spaces (e.g. Atari), not directly for continuous control.
* **Double DQN, Dueling DQN**

  * Variants to reduce overestimation bias and improve stability.

*Robotics note:* Discretizing your control can work but often loses precision and makes training slow.

---

## 2. Model-Free, Policy-Gradient

These directly parameterize a policy $\pi_\theta(a\mid s)$ and optimize expected return via gradients.

* **REINFORCE**

  * The simplest “Monte-Carlo” policy gradient.
  * High variance; poor sample efficiency.
* **TRPO (Trust Region PG)**

  * Constrains policy updates to stay within a “trust region.”
  * Stable but computationally heavy.
* **PPO (Proximal PG)**

  * A faster, simpler approximation to TRPO using clipped objectives.
  * Widely used in robotics for its stability vs. simplicity trade-off.

*Robotics note:* On-policy methods like PPO/TRPO are stable but need lots of samples—often done in simulation then transferred.

---

## 3. Model-Free, Actor-Critic

Combine a policy (actor) with a value estimator (critic) to reduce variance and improve sample efficiency.

* **A2C / A3C**

  * Synchronous (A2C) or asynchronous (A3C) parallel workers.
  * Good for scaling but still on-policy.
* **DDPG (Deep Deterministic PG)**

  * Off-policy; handles continuous actions by learning a deterministic policy.
  * Prone to hyperparameter sensitivity and stability issues.
* **TD3 (Twin Delayed DDPG)**

  * Fixes DDPG’s overestimation by using two critics and delayed updates.
  * More stable for robotics tasks.
* **SAC (Soft Actor-Critic)**

  * Off-policy and maximum-entropy objective for better exploration.
  * Currently one of the best all-around choices for continuous control in robotics.

*Robotics note:* Off-policy actor-critic methods (TD3, SAC) tend to be sample-efficient and robust enough to train real-robot controllers.

---

## 4. Model-Based RL

These explicitly learn (or are given) a dynamics model $f_\phi(s,a)$ and use it to plan or generate synthetic data.

* **PILCO**

  * Gaussian-process dynamics; extremely sample efficient but scales poorly.
* **MBPO (Model-Based Policy Optimization)**

  * Learns a neural dynamics model and interleaves policy updates on both real and imagined rollouts.
* **Dreamer, PETS**

  * Use latent-variable models or ensemble planning for long-horizon tasks.

*Robotics note:* Model-based approaches can train in tens to hundreds of real-world episodes—but require careful model-error management.

---

## 5. Imitation & Inverse RL

Leverage expert demonstrations to speed up learning or infer reward functions.

* **Behavioral Cloning (BC)**

  * Supervised learning on $(s,a)$ pairs.
  * Quick bootstrap but can drift if the policy strays from the expert distribution.
* **DAgger**

  * Dataset aggregation to correct drift by querying the expert on visited states.
* **GAIL (Generative Adversarial IL)**

  * Adversarially matches expert state-action distributions; infers an implicit reward.

*Robotics note:* When you have human demonstrations (e.g. teleoperation), imitation learning is often combined with RL fine-tuning.

---

## 6. Hierarchical & Meta-RL

Decompose long-horizon tasks or learn to adapt quickly.

* **Options / Feudal Networks**

  * High-level “options” or sub-policies that invoke lower-level controllers.
* **MAML, RL^2**

  * Meta-learning frameworks that aim for rapid adaptation to new tasks.

*Robotics note:* Hierarchical controllers can mirror the natural modularity of robot tasks (e.g. “grasp,” “move,” “place”).

---

## 7. Evolutionary & Black-Box Methods

Optimize policies as black-box functions, often via population-based search.

* **CMA-ES, NES**

  * Simple to parallelize and explore, but typically very sample inefficient.

*Robotics note:* Good for rapidly prototyping or for tasks where gradients are very noisy, but usually relegated to simulation.

---

### Choosing for Robotics

* **Continuous Control** → focus on **DDPG/TD3/SAC** (off-policy actor-critic) or **PPO** (on-policy, if you can afford simulation samples).
* **Sample Efficiency** → consider **model-based** (MBPO, PILCO) or **imitation + RL hybrid**.
* **Stability** → **PPO** or **SAC** are battle-tested.
* **Complex Tasks** → layer on **hierarchical** frameworks or **meta-learning** if you need rapid task switching or structured policies.
