# Learn-RL 🤖🏔️

A clean, modular repository tracking my journey learning Reinforcement Learning (RL) through hands-on implementations in modern 3D physics simulators. The project transitions from basic 2D-adjacent balancing tasks to high-dimensional continuous robotic control and coordinate tracking.

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.11+
* **Physics Simulator:** [Gymnasium MuJoCo](https://farama.org) (Native open-source `v5` environments)
* **RL Framework:** [Stable-Baselines3](https://readthedocs.io)
* **Core Algorithms:** PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic)
* **Hardware Target:** CPU for low-dimensional vector-observation environments (01–05); GPU (CUDA) for the wide 512×512 networks used in the Humanoid experiments (06), where vectorized parallel environments make CPU core count the real throughput bottleneck.

---

## 📂 Repository Structure
```text
Learn-RL/
│
├── 3d_mujoco/                        # Core executable Python training scripts
│   ├── 01_cartpole.py                 # 3D Inverted Pendulum Balance
│   ├── 02_half_cheetah.py             # Locomotion with Orientation Wrapper
│   ├── 03_hopper.py                   # Stabilized Single-Legged Hopping (gSDE)
│   ├── 04_ant.py                      # Quadruped Coordination (8 Continuous Joints)
│   ├── 05_reacher.py                  # Robotic Arm Inverse Kinematics Tracking
│   ├── 06_humanoid_baseline.py        # 17-Joint Bipedal Locomotion: raw-speed baseline
│   ├── 06_humanoid_curriculum.py      # Stage 1 (speed) -> Stage 2 (+ physics efficiency)
│   ├── 06_humanoid_sprinter.py        # Fine-tuned: speed-maximizing variant
│   └── 06_humanoid_marathoner.py      # Fine-tuned: efficiency-maximizing variant
│
├── archive/                          # Superseded early approaches, kept for reference
│   ├── 06_humanoid_tall_posture.py    # v0: hard-coded upright posture constraint
│   └── 06_humanoid_symmetric_gait.py  # v0: bilateral-symmetry reward penalty
│
├── models/                           # Untracked artifacts directory for saved weights
│   ├── ppo_mujoco_3d_cartpole.zip
│   ├── sac_mujoco_cheetah.zip
│   ├── best_sac_mujoco_hopper.zip
│   ├── best_sac_mujoco_ant.zip
│   ├── best_sac_mujoco_reacher.zip
│   ├── sac_humanoid_stage1.zip         # Curriculum Stage 1 (speed-only)
│   ├── sac_humanoid_stage2.zip         # Curriculum Stage 2 (+ efficiency fine-tune)
│   ├── sac_humanoid_sprinter.zip       # Speed-specialized fine-tune
│   ├── sac_humanoid_marathoner.zip     # Efficiency-specialized fine-tune
│   └── vecnormalize_*.pkl              # Observation/reward normalization stats per model
│
└── .gitignore                        # Prevents tracking heavy binary weights (.zip)
```

---

## 🏆 Project Progression & Milestones

### 01. 3D CartPole (`InvertedPendulum-v5`)
* **Algorithm:** PPO (`MlpPolicy`)
* **Objective:** Keep a vertical pole perfectly balanced on a moving cart.
* **Key Learnings:** Overcoming environment version shifts and scaling test limit capabilities past default horizons (extended successfully to a `5,000` step custom limit).
* **Result:** **Solved.** Perfect sustained balance without falling.

### 02. Upright HalfCheetah (`HalfCheetah-v5`)
* **Algorithm:** SAC (`MlpPolicy`)
* **Objective:** Drive a two-legged robot forward as fast as possible.
* **The Challenge:** The agent originally collapsed into local minimum traps, sliding on its stomach or back to collect lazy baseline speed points.
* **The Solution:** Engineered a custom `UprightCheetahWrapper` that intercepts the MuJoCo position matrix (`qpos`) and forces a hard episode termination + penalty if the torso tilts past $\approx 80^\circ$.
* **Result:** **Success.** Broke out of the local minimum. Sprinted upright to a massive score of **`1.06e+04`** (10,600+).

### 03. Stabilized Hopper (`Hopper-v5`)
* **Algorithm:** SAC + gSDE
* **Objective:** Command a single-legged pogo-stick robot to jump forward cleanly.
* **The Challenge:** The agent suffered from severe late-stage policy degradation, dropping from a peak of `2,400` down to `1,930` due to noisy joint actions causing crash landings.
* **The Solution:**
  1. Implemented **State-Dependent Exploration (`use_sde=True`)** to mimic real-world continuous physics disturbances instead of spastic millisecond twitches.
  2. Set up an automated **`EvalCallback` Checkpoint System** hooked to local file renaming scripts to ignore decayed end-states and load the peak-performing network array.
* **Result:** **Perfect.** Cleared all visual evaluation runs up to the maximum 1,000-step ceiling without tipping over.

### 04. The Ant crawler (`Ant-v5`)
* **Algorithm:** SAC + Checkpoints + gSDE
* **Objective:** Coordinate **8 independent continuous joint motors** (2 per leg) to make a quadruped crawl forward.
* **Key Learnings:** Managing large action and observation spaces. Managing extended structural exploration stages while the network builds its initial coordination buffer.
* **Result:** **Elite.** Achieved an outstanding score of **`5.02e+03`** (5,020+), running clean error-free 1,000-step loops.

### 05. Robotic Arm Target Tracker (`Reacher-v5`)
* **Algorithm:** SAC + gSDE
* **Objective:** Coordinate a multi-joint arm to dynamically position its fingertip inside a randomly spawning target coordinate point.
* **Key Learnings:** Overcoming distance-to-target penalty mathematics. An untrained arm scores -50 to -30, but a solved network scores near 0.
* **Result:** **Solved.** Achieved a spectacular evaluation score of **`-2.1`**, snapping onto goals instantly with zero muscle twitching.

### 06. The 17-Joint Humanoid (`Humanoid-v5`)
* **Algorithm:** SAC + VecNormalize + Checkpoints + gSDE (Deep 512×512 Architecture, vectorized parallel envs)
* **Objective:** Evolved over several iterations — from "walk without falling," to "run as fast as possible with zero human-motion bias (per the Bitter Lesson)," to "run fast *and* be physically efficient (real mechanical power, Cost of Transport, friction-slip loss)."
* **v0 attempts (archived):** Early versions hard-coded human-motion priors — a minimum "tall posture" height and a bilateral leg-symmetry penalty. Both were abandoned once the project's philosophy shifted toward letting the physics, not human anatomy, define what "good" locomotion looks like.
* **The Baseline (`06_humanoid_baseline.py`):** A minimal-bias reward (bounded forward velocity + alive bonus + light control/smoothness costs, no gait shaping) reliably produced a **sustained ~3+ m/s gait for the full 1,000-step episode** — proof that raw speed alone is a learnable, stable objective for this environment.
* **The Challenge — reward hacking, three different ways:** Adding real-physics efficiency terms (mechanical power, Cost of Transport, friction-slip energy loss) on top of the baseline broke training three separate times, each in a different way:
  1. **The "stand still forever" exploit** — a flat per-step alive bonus plus a Cost-of-Transport calculation that used a velocity floor (to avoid divide-by-zero) accidentally made near-zero movement look "efficient," so the agent just stood there for all 1,000 steps.
  2. **The "dive and die" exploit** — after fixing #1, the power/friction penalty weights were large enough that surviving each additional step had *negative* expected value; the agent learned to fall over almost immediately rather than accumulate more per-step losses.
  3. **The "free effort past the cap" exploit** — a hard `min()` ceiling on the power/friction penalties created a zero-marginal-cost region once the cap was hit, so the agent maxed out torque (~900W+) for no extra penalty, producing violent, short-lived flailing.
* **The Solution:**
  1. Replaced hard `min()` penalty caps with **`tanh`-based soft saturation** — bounded like a cap, but the marginal cost of "more effort" never fully vanishes.
  2. Rebalanced the alive bonus and efficiency-term weights so per-step reward stays net-positive under reasonable behavior, not just under standing still or dying quickly.
  3. Adopted a **curriculum / warm-start strategy**: train Stage 1 (speed-only, proven-stable reward) to convergence first, then load that checkpoint and fine-tune with small efficiency weights turned on — rather than trying to learn locomotion and efficiency simultaneously from a fresh network. Re-enabling entropy (`ent_coef="auto"`) on load was necessary, since Stage 1's exploration noise anneals to near-zero by the end of training and would otherwise prevent the policy from adapting to new reward terms at all.
* **Result:** **In active refinement.** The curriculum approach (`06_humanoid_curriculum.py`) produces a stable gait that is both faster and more efficient (lower Cost of Transport) than the Stage 1 baseline. Two specialized fine-tunes now explore opposite ends of the speed/efficiency tradeoff: `06_humanoid_sprinter.py` (speed weighted heavily, efficiency barely constrains effort) and `06_humanoid_marathoner.py` (efficiency weighted heavily, speed is secondary). The resulting gait is intentionally **asymmetric and uses both knee and foot ground contact** — left as-is, since no bilateral-symmetry or "look human" prior was ever part of the design goal.

---

## ⚡ Key Engineering Insights

### 1. The Host-to-Device Memory Transfer Bottleneck
During early testing (01–05), forcing a tiny Multi-Layer Perceptron (MLP) neural network onto a high-performance GPU (`cuda`) actually *increased* step processing latency compared to standard CPU execution. Because the 3D physics runs sequentially on the CPU, the overhead of shifting microscopic vector data blocks across the motherboard's PCIe lanes slows down training frames-per-second (FPS). **Lesson:** for small networks and single environments, keep vector spaces on the CPU; save the GPU for massive image frameworks (`CnnPolicy`) — or, as with 06, for wide networks trained across many *vectorized parallel* environments, where GPU throughput starts to matter again and CPU core count becomes the bottleneck for physics stepping instead.

### 2. PPO vs. SAC in Complex Environments
While PPO handles simple balancing setups beautifully, its "on-policy" nature makes it rigid and prone to falling into lazy local minimum traps when controlling complex multi-joint bodies. Switching to Soft Actor-Critic (SAC) introduced an automated entropy framework that rewards the agent for staying curious, making it the superior choice for robotic locomotion.

### 3. Reward Hacking Is the Default Outcome, Not an Edge Case
Every one of the three Humanoid efficiency-reward failures (stand-still, dive-and-die, free-effort-past-cap) was the policy correctly and rationally exploiting exactly what the reward function said, not a training bug. **Lesson:** when a policy converges to an unwanted degenerate behavior, the reward math is the first thing to audit, not the training duration — more steps just makes the agent better at exploiting whatever the reward actually incentivizes.

### 4. Hard Caps vs. Soft Caps on Penalty Terms
A hard `min(cost, cap)` ceiling on a penalty term creates a region with **zero marginal cost** once the cap is saturated — effectively telling the agent "anything beyond this point is free." A smooth saturating function (e.g., `cap * tanh(raw / cap)`) bounds the penalty just as effectively but never fully flattens the incentive to reduce it further. **Lesson:** bound penalty terms with smooth saturation, not hard clipping.

### 5. Curriculum / Warm-Start Fine-Tuning for Multi-Objective Rewards
Trying to learn locomotion and physical efficiency simultaneously from a randomly initialized network made every efficiency-term failure mode (above) much easier to fall into. Training a simple, proven objective to convergence first, then loading that checkpoint to fine-tune additional objectives at small weights, was far more reliable. **Lesson:** for multi-term rewards, get one term working well in isolation before stacking on the next — and remember to re-enable exploration (reset/raise the entropy coefficient) when fine-tuning a converged checkpoint on new reward terms, or the policy won't have enough exploration noise left to adapt.

---

## 🚀 How to Run Locally

1. **Clone the Repo:**
   ```bash
   git clone https://github.com
   cd Learn-RL/3d_mujoco
   ```
2. **Setup Dependencies:**
   Ensure you are using **Python 3.11** or **3.12**:
   ```bash
   pip install gymnasium[mujoco] stable-baselines3
   ```
   For the Humanoid scripts (06), also install a CUDA-enabled PyTorch build matching your GPU driver version (see PyTorch's official install selector) — training these will attempt to use `device="cuda"`.
3. **Run Any Project:**
   ```bash
   python -u 06_humanoid_baseline.py
   ```
   *(Press **Tab** in the pop-up 3D window to activate automated camera tracking!)*

   For the Humanoid curriculum, run `06_humanoid_curriculum.py` with `STAGE = 1` first, confirm the resulting gait via the built-in visual test, then set `STAGE = 2` and rerun to fine-tune in physics-efficiency terms. `06_humanoid_sprinter.py` and `06_humanoid_marathoner.py` each fine-tune further from the Stage 2 checkpoint toward opposite ends of the speed/efficiency tradeoff.