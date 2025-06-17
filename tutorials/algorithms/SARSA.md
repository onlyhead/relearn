# SARSA: On-Policy Temporal Difference Learning

## Introduction

SARSA (State-Action-Reward-State-Action) is a fundamental on-policy reinforcement learning algorithm that learns the value of the policy being followed, rather than the optimal policy. Named after the quintuple of information it uses for updates (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}), SARSA represents a more conservative approach to learning compared to its off-policy cousin, Q-Learning.

## Running Example: The Windy Grid World

We'll use a challenging windy grid world to illustrate SARSA's behavior:

```
Wind strength: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  S  │     │     │ ↑   │ ↑   │ ↑   │ ↑↑  │ ↑↑  │ ↑   │  G  │  Row 3
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│     │     │     │ ↑   │ ↑   │ ↑   │ ↑↑  │ ↑↑  │ ↑   │     │  Row 2
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│     │     │     │ ↑   │ ↑   │ ↑   │ ↑↑  │ ↑↑  │ ↑   │     │  Row 1
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│     │     │     │ ↑   │ ↑   │ ↑   │ ↑↑  │ ↑↑  │ ↑   │     │  Row 0
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  0     1     2     3     4     5     6     7     8     9

S = Start (3,0), G = Goal (3,7)
Wind pushes agent upward: ↑ = 1 cell, ↑↑ = 2 cells
Actions: ↑ (Up), ↓ (Down), ← (Left), → (Right)
Reward: -1 per step, 0 at goal
```

**The Challenge**: Wind makes the environment stochastic from the agent's perspective!

## The Core Concept

### What is SARSA?

SARSA is an **on-policy** temporal difference learning algorithm that learns the action-value function Q^π(s,a) for the policy π currently being followed. Unlike Q-Learning, which learns about the optimal policy regardless of the behavior policy, SARSA learns about the actual policy being executed.

### The SARSA Quintuple

Every SARSA update uses five pieces of information:

```
Time Step Sequence:
    t           t+1         t+2
┌─────────┐  ┌─────────┐  ┌─────────┐
│   S_t   │  │  S_t+1  │  │  S_t+2  │
│   A_t   │  │  A_t+1  │  │   ...   │
└─────────┘  └─────────┘  └─────────┘
      │           │
      │    R_t+1  │
      └───────────┘

SARSA Quintuple: (S_t, A_t, R_t+1, S_t+1, A_t+1)
                  ↑    ↑     ↑      ↑      ↑
               State Action Reward Next   Next
               now   now    got   State  Action
```

### SARSA vs Q-Learning: The Key Difference

```
Situation: Agent at (2,5) in windy grid

Q-Learning Update:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                        ↑
                   Uses BEST possible action
                   (regardless of what agent will actually do)

SARSA Update:
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
                        ↑
                   Uses ACTUAL next action
                   (the action agent will really take)

Practical Impact:
┌─────────────────────────────┐    ┌─────────────────────────────┐
│         Q-Learning          │    │           SARSA             │
│      (Off-Policy)           │    │        (On-Policy)          │
├─────────────────────────────┤    ├─────────────────────────────┤
│ "What's the best I could    │    │ "What will I actually get   │
│  do from the next state?"   │    │  from the next state given  │
│                             │    │  my current policy?"        │
│ → Optimistic               │    │ → Realistic                 │
│ → Risk-seeking             │    │ → Risk-aware               │
└─────────────────────────────┘    └─────────────────────────────┘
```

## The SARSA Algorithm in Detail

### The Update Rule Breakdown

```
SARSA Update Formula:
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_t+1 + γQ(S_t+1, A_t+1) - Q(S_t, A_t)]

Component Analysis:
┌─────────────────────────────────────────────────────────────────┐
│                     SARSA Update Anatomy                       │
├─────────────────────────────────────────────────────────────────┤
│ Q(S_t, A_t)     │ Current estimate of action value            │
│ α               │ Learning rate (how much to update)          │
│ R_t+1           │ Immediate reward received                    │
│ γ               │ Discount factor (future reward importance)   │
│ Q(S_t+1, A_t+1) │ Next state-action value (ACTUAL next action)│
│ TD Error        │ R_t+1 + γQ(S_t+1, A_t+1) - Q(S_t, A_t)     │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Example in Windy Grid

```
Episode Trace:
Step 1: Agent at (3,0), chooses → (ε-greedy with ε=0.1)
Step 2: Wind = 0, agent moves to (3,1), chooses → again
Step 3: Wind = 0, agent moves to (3,2), chooses → again  
Step 4: Wind = 1, agent tries → but ends up at (2,3), chooses ↓
Step 5: Wind = 1, agent tries ↓ but ends up at (2,4), chooses →

Let's trace the SARSA update for Step 4:

Before Step 4:
┌─────────────────────────────────────────────────────────────────┐
│ State (3,2): Q-values                                           │
│ ↑: -15.2  ↓: -12.8  ←: -18.1  →: -11.5  ← Current best        │
└─────────────────────────────────────────────────────────────────┘

Step 4 Execution:
S_t = (3,2), A_t = →, Agent tries to go right...
Wind pushes up! Actual result: S_t+1 = (2,3)
R_t+1 = -1 (step cost)
A_t+1 = ↓ (chosen by ε-greedy from state (2,3))

SARSA Update:
Q((3,2), →) ← Q((3,2), →) + α[R_t+1 + γQ((2,3), ↓) - Q((3,2), →)]
Q((3,2), →) ← -11.5 + 0.1[-1 + 0.9×(-13.2) - (-11.5)]
Q((3,2), →) ← -11.5 + 0.1[-1 - 11.88 + 11.5]
Q((3,2), →) ← -11.5 + 0.1[-1.38]
Q((3,2), →) ← -11.5 - 0.138 = -11.638

After Step 4:
┌─────────────────────────────────────────────────────────────────┐
│ State (3,2): Q-values                                           │
│ ↑: -15.2  ↓: -12.8  ←: -18.1  →: -11.638  ← Updated!          │
└─────────────────────────────────────────────────────────────────┘
```

### SARSA Pseudocode

```
Algorithm: SARSA
Initialize Q(s,a) arbitrarily ∀s ∈ S, a ∈ A(s)
For each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., ε-greedy)
    For each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., ε-greedy)
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        S ← S'; A ← A'
    Until S is terminal
```

## On-Policy vs Off-Policy: The Philosophical Difference

### Policy Consistency in SARSA

```
SARSA's On-Policy Nature:

┌─────────────────────────────────────────────────────────────────┐
│                    The Policy Loop                             │
│                                                                 │
│  Current Policy π  ──────────────────────┐                     │
│         │                                │                     │
│         ↓                                │                     │
│  Choose Actions A_t, A_t+1              │                     │
│         │                                │                     │
│         ↓                                │                     │
│  Experience: (S_t, A_t, R_t+1, S_t+1, A_t+1)                 │
│         │                                │                     │
│         ↓                                │                     │
│  Update Q(S_t, A_t) using A_t+1          │                     │
│         │                                │                     │
│         ↓                                │                     │
│  Improve Policy π from Q  ───────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Key Insight: SARSA learns about the policy it's actually following!
```

### Risk-Aware Learning Example

```
Dangerous Cliff Environment:

┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  S  │     │     │     │     │     │     │     │     │     │     │  G  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│     │     │     │     │     │     │     │     │     │     │     │     │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│     │     │     │     │     │     │     │     │     │     │     │     │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │ 💀  │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
💀 = Cliff (-100 reward)

Q-Learning Behavior:
"I know the cliff edge path is risky with ε-greedy exploration,
 but the OPTIMAL policy would take that path.
 So I'll learn high Q-values for the cliff edge."

SARSA Behavior:
"I'm using ε-greedy, so I might accidentally step off the cliff.
 Given MY policy, the cliff edge is dangerous.
 I'll learn to prefer the safer path."

Learned Policies:
┌─────────────────────────────┐    ┌─────────────────────────────┐
│        Q-Learning           │    │          SARSA              │
│     (Cliff-Edge Path)       │    │      (Safe Path)            │
├─────────────────────────────┤    ├─────────────────────────────┤
│ S→→→→→→→→→→→G               │    │ S                           │
│                             │    │ ↓                           │
│                             │    │ ↓                           │
│                             │    │ →→→→→→→→→→↑                 │
│ 💀💀💀💀💀💀💀💀💀💀💀     │    │ 💀💀💀💀💀💀💀💀💀💀💀     │
│                             │    │                             │
│ Optimal but risky!          │    │ Suboptimal but safe!        │
└─────────────────────────────┘    └─────────────────────────────┘
```

## Exploration Strategies in SARSA

### ε-Greedy with SARSA

```
SARSA + ε-Greedy Decision Process:

State S_t: Current position
    │
    ↓
ε-Greedy Action Selection for A_t
    │
    ├─── ε probability ────→ Random Action
    │                           │
    └── (1-ε) probability ──→ argmax_a Q(S_t,a)
                                │
    ┌───────────────────────────┘
    │
    ↓
Execute A_t, observe R_t+1, S_t+1
    │
    ↓
ε-Greedy Action Selection for A_t+1  ← Critical: Use SAME policy!
    │
    ↓
SARSA Update: Q(S_t,A_t) += α[R_t+1 + γQ(S_t+1,A_t+1) - Q(S_t,A_t)]
```

### Policy Improvement and Convergence

```
SARSA Convergence Process:

Episode 1-50: High Exploration (ε = 0.3)
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ 🤖  │ 🎲  │ 🎲  │ ↑🎲 │ ↑🎲 │ ↑🎲 │ ↑↑🎲│ ↑↑🎲│ ↑🎲 │  G  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 🎲  │ 🎲  │ 🎲  │ ↑🎲 │ ↑🎲 │ ↑🎲 │ ↑↑🎲│ ↑↑🎲│ ↑🎲 │ 🎲  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
🎲 = Frequent random actions, high uncertainty

Episode 200-300: Medium Exploration (ε = 0.1)
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ 🤖→ │ →→  │ →→  │ ↑→  │ ↑→  │ ↑→  │ ↑↑→ │ ↑↑→ │ ↑→  │  G  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑↑  │ ↑↑  │ ↑   │ ↑   │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Clearer preferences emerging, but still some exploration

Episode 500+: Low Exploration (ε = 0.05)
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ 🤖→ │ →→  │ →→  │ ↑→  │ ↑→  │ ↑→  │ ↑↑→ │ ↑↑→ │ ↑→  │  G  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑   │ ↑↑  │ ↑↑  │ ↑   │ ↑   │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Stable policy adapted to wind patterns
```

## Advanced SARSA Variants

### SARSA(λ) - Eligibility Traces

SARSA can be enhanced with eligibility traces for faster learning:

```
Eligibility Traces in SARSA:

Standard SARSA (λ=0):
Episode: (3,0)→(3,1)→(3,2)→(2,3)→(1,4) [Wind effects]
Only updates: Q((3,2), →) based on immediate successor

SARSA(λ) with λ=0.8:
Episode: (3,0)→(3,1)→(3,2)→(2,3)→(1,4)

Eligibility traces after visiting (3,2), →:
e((3,2), →) = 1.0      ← Current state-action (full credit)
e((3,1), →) = 0.8      ← Previous step (80% credit)
e((3,0), →) = 0.64     ← Two steps back (64% credit)

Update Distribution:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│64%  │ 80% │100% │     │     │     │     │     │     │  G  │  
│ ↑   │  ↑  │  ⭐ │     │     │     │     │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
All recent states get updated proportionally!

Benefits:
• Faster learning from successful episodes
• Better credit assignment
• More efficient in environments with long action sequences
```

### Expected SARSA - The Best of Both Worlds

Expected SARSA modifies the update to use expected values:

```
Traditional SARSA:
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                          ↑
                    Uses actual next action

Expected SARSA:
Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S') Q(S',a) - Q(S,A)]
                          ↑
                    Uses expected value over next actions

Visual Comparison:
┌─────────────────────────────────────────────────────────────┐
│             Next State S': Q-values                        │
│        ↑: 2.1   ↓: 1.8   ←: 1.2   →: 2.5                  │
│                                                             │
│ SARSA: If A' = ↓, use Q(S',↓) = 1.8                       │
│                                                             │
│ Expected SARSA: Use π(↑)×2.1 + π(↓)×1.8 + π(←)×1.2 + π(→)×2.5│
│ With ε-greedy (ε=0.1): 0.025×2.1 + 0.025×1.8 + 0.025×1.2 + 0.925×2.5│
│                      = 0.0525 + 0.045 + 0.03 + 2.3125 = 2.44│
└─────────────────────────────────────────────────────────────┘

Benefits:
• Reduces variance in updates
• Often converges faster than regular SARSA
• More stable learning
```

## Practical Considerations

### Learning Rate Scheduling

```
SARSA Learning Rate Schedules:

Constant Learning Rate (α = 0.1):
0.1 ─┬─────────────────────────────────────→
     │ • Simple to implement
     │ • May not converge optimally
0.0 ─┴─────────────────────────────────────→ episodes

Decaying Learning Rate (α = 1/(t+1)):
1.0 ─┐
     │ ╲
0.5 ─┤  ╲___
     │      ╲____
     │           ╲_____
0.0 ─┴─────────────────╲________________→ episodes
     • Guaranteed convergence (under conditions)
     • Slower adaptation to changes

State-based Learning Rate:
α(s,a) = α₀ / (visits(s,a) + 1)
     • Higher learning rate for rarely visited states
     • Lower learning rate for well-explored states
     • Adaptive to exploration patterns
```

### Action Selection Policies

```
Common SARSA Action Selection Policies:

1. ε-Greedy (most common):
   P(a) = {  1-ε+ε/|A|  if a = argmax Q(s,a)
          {  ε/|A|      otherwise

2. Boltzmann/Softmax:
   P(a) = e^(Q(s,a)/τ) / Σ_a' e^(Q(s,a')/τ)
   
   Temperature Evolution:
   High τ (τ=2.0): More uniform action selection
   ┌─────────────────────────────────────────┐
   │ ↑: ████████   ↓: ███████   ←: ██████    │
   │ →: █████████                            │
   └─────────────────────────────────────────┘
   
   Low τ (τ=0.1): More greedy action selection  
   ┌─────────────────────────────────────────┐
   │ ↑: ██        ↓: █         ←: ▌          │
   │ →: ████████████████████████████████████ │
   └─────────────────────────────────────────┘

3. Upper Confidence Bound (UCB):
   Select: argmax_a [Q(s,a) + c√(ln(N(s))/N(s,a))]
   Balances exploitation with exploration of uncertain actions
```

## Strengths and Limitations

### Strengths of SARSA

```
✓ Conservative Learning:
  ┌─────────────────────────────────────────┐
  │ Learns safe policies that account for   │
  │ exploration risks                       │
  └─────────────────────────────────────────┘

✓ Policy Consistency:
  ┌─────────────────────────────────────────┐
  │ Evaluates the policy actually being     │
  │ followed, not an idealized policy       │
  └─────────────────────────────────────────┘

✓ Guaranteed Convergence:
  ┌─────────────────────────────────────────┐
  │ Under standard conditions, converges    │
  │ to optimal policy for the given         │
  │ exploration strategy                    │
  └─────────────────────────────────────────┘

✓ Real-world Safety:
  ┌─────────────────────────────────────────┐
  │ Better for applications where           │
  │ exploration mistakes are costly         │
  └─────────────────────────────────────────┘
```

### Limitations of SARSA

```
✗ Slower Convergence:
  ┌─────────────────────────────────────────┐
  │ May take longer to find optimal policy  │
  │ compared to Q-Learning                  │
  └─────────────────────────────────────────┘

✗ Policy Dependence:
  ┌─────────────────────────────────────────┐
  │ Performance heavily depends on          │
  │ exploration strategy choice             │
  └─────────────────────────────────────────┘

✗ Suboptimal Policies:
  ┌─────────────────────────────────────────┐
  │ May converge to suboptimal policies     │
  │ if exploration never stops              │
  └─────────────────────────────────────────┘

✗ Exploration-Exploitation Tradeoff:
  ┌─────────────────────────────────────────┐
  │ Must carefully balance exploration      │
  │ to maintain both learning and safety    │
  └─────────────────────────────────────────┘
```

## Applications and Use Cases

### When to Choose SARSA

```
Ideal SARSA Applications:

🤖 Robotics:
   • Physical robots where exploration mistakes are expensive
   • Safe navigation in uncertain environments
   • Human-robot interaction scenarios

🚗 Autonomous Vehicles:
   • Learning driving policies where safety is paramount
   • Adaptive cruise control systems
   • Lane-keeping assistance

🏭 Industrial Control:
   • Process optimization where failures are costly
   • Resource allocation with safety constraints
   • Quality control systems

🎮 Game AI (specific cases):
   • Games where conservative play is advantageous
   • Multi-player environments with adaptation
   • Real-time strategy with risk management

📈 Finance:
   • Portfolio optimization with risk constraints
   • Algorithmic trading with downside protection
   • Resource allocation under uncertainty
```

### SARSA vs Q-Learning Decision Matrix

```
Environment Characteristics Decision Guide:

┌─────────────────────────────────────────────────────────────────┐
│                    Choose SARSA when:                          │
├─────────────────────────────────────────────────────────────────┤
│ • Exploration mistakes are costly                               │
│ • Conservative policies preferred                               │
│ • Policy consistency important                                  │
│ • Real-world safety critical                                   │
│ • Stochastic environment with risks                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Choose Q-Learning when:                      │
├─────────────────────────────────────────────────────────────────┤
│ • Fast convergence to optimal policy desired                   │
│ • Exploration mistakes have low cost                           │
│ • Sample efficiency is critical                                │
│ • Offline learning from stored experiences                     │
│ • Maximum performance regardless of exploration                │
└─────────────────────────────────────────────────────────────────┘
```

## Mathematical Foundations

### Convergence Theory

SARSA convergence relies on the same mathematical foundations as Q-Learning, but with additional considerations for policy consistency:

```
Convergence Conditions for SARSA:

1. Policy Improvement Condition:
   The policy π must improve based on current Q-values
   π'(s) = argmax_a Q^π(s,a)

2. Exploration Condition:
   All state-action pairs must be visited infinitely often
   lim_{t→∞} N_t(s,a) = ∞ ∀s,a

3. Learning Rate Conditions:
   Σ_{t=0}^∞ α_t = ∞  and  Σ_{t=0}^∞ α_t² < ∞

4. Policy Convergence:
   The exploration probability must decay: lim_{t→∞} ε_t = 0

Convergence Guarantee:
If all conditions are met, SARSA converges to Q*, the optimal 
action-value function, and the policy converges to π*.
```

## Conclusion

SARSA represents a fundamental approach to on-policy reinforcement learning that prioritizes safety and policy consistency over pure optimality. Its conservative nature makes it particularly valuable in real-world applications where exploration mistakes carry significant costs.

While Q-Learning asks "What's the best I could possibly do?", SARSA asks "What will I realistically achieve given how I actually behave?" This distinction makes SARSA indispensable for applications requiring safe, reliable learning in uncertain environments.

The algorithm's elegance lies in its simplicity: by learning about the policy being followed rather than an idealized optimal policy, SARSA provides a more honest assessment of expected performance, leading to safer and more reliable behavior in critical applications.

Understanding SARSA is essential for any practitioner of reinforcement learning, as it illuminates the fundamental tradeoffs between optimality and safety, and between idealized and realistic performance evaluation.
