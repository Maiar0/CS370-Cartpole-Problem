# CS370 - CartPole Problem

## Purpose

This project demonstrates the application of reinforcement learning algorithms—specifically, **REINFORCE** and **Advantage Actor-Critic (A2C)**—to solve the classic CartPole problem from the OpenAI Gym environment. The objective is to train an agent to balance a pole on a moving cart by learning optimal policies through trial-and-error interactions with the environment.

## Implementation

The implementation uses **Keras** and **TensorFlow** to build and train policy networks. Two distinct policy-gradient approaches are included:

---

### **REINFORCE Algorithm**

The REINFORCE method is a Monte Carlo policy gradient algorithm. It updates the policy at the end of each episode based on the total return received.

**Key Steps:**

```Courier New
Initialize policy parameters θ
Repeat for each episode:
    Initialize state s₀
    Repeat for each time step:
        Select action aₜ from policy π(a|sₜ, θ)
        Execute action aₜ, observe reward rₜ and next state sₜ₊₁
        Store (sₜ, aₜ, rₜ)
    End episode
    Compute return G for each time step
    Update θ ← θ + α ∑ G ∇θ log π(a|sₜ, θ)
```

---

### **Advantage Actor-Critic (A2C) Algorithm**

A2C improves upon REINFORCE by using both a policy (actor) and a value function (critic) to reduce variance and improve learning stability. The actor updates occur at every time step based on an advantage estimate.

**Key Steps:**

```Courier New
Initialize actor and critic parameters (θₐ, θᵥ)
Repeat for each episode:
    Initialize state s₀
    Repeat for each time step:
        Select action aₜ from π(a|sₜ, θₐ)
        Execute action aₜ, observe reward rₜ and next state sₜ₊₁
        Compute advantage Aₜ = rₜ + γ V(sₜ₊₁, θᵥ) - V(sₜ, θᵥ)
        Update actor: θₐ ← θₐ + α Aₜ ∇θ log π(aₜ|sₜ, θₐ)
        Update critic: θᵥ ← θᵥ + β ∇θᵥ (Aₜ)²
```

---

### **Score Logging and Evaluation**

A custom `ScoreLogger` class tracks and logs agent performance over episodes:

- Saves per-episode scores in CSV format
- Computes rolling average over 100 episodes
- Determines when the environment is "solved" (average score ≥ 195 over 100 episodes)
- Prints min, average, and max scores for diagnostic purposes

CSV output includes:
- `scores.csv`: Raw per-episode scores
- `solved.csv`: Run index when the environment was solved

---

This project showcases practical use of deep reinforcement learning methods, emphasizing the differences in learning stability, convergence speed, and variance between policy-gradient and actor-critic strategies.
