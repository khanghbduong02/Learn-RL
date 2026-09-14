"""
SPRINTER (v2): automated exploration for maximum achievable speed, rather
than manually guessing speed_weight one value at a time.

Two phases, controlled by MODE below:
  MODE = "sweep": trains several speed_weight values briefly (SWEEP_STEPS
                  each), evaluates each deterministically, logs results to
                  sprinter_sweep_results.csv, and prints a ranked summary.
                  Nothing here is a final answer -- it's data to pick from.
  MODE = "final": trains the winning config (set WINNING_SPEED_WEIGHT
                  below after reviewing the sweep CSV) for a much longer
                  budget, optionally across multiple seeds, keeping
                  whichever seed's final eval velocity is highest.

REAL PHYSICS NOTE: max_speed_cap is set very high (50.0) so the reward
never artificially ceilings velocity -- the actual ceiling is whatever
this body's actuator torque limits and gear ratios (fixed in the
Humanoid-v5 MuJoCo model, not something this script changes) allow. Power
is computed directly from MuJoCo's qfrc_actuator * qvel (real Watts), not
a proxy.
"""
import os
import csv
import time
import warnings
import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

GRAVITY = 9.81

# ============================================================================
MODE = "final"  # "sweep" or "final"

SWEEP_SPEED_WEIGHTS = [2.0, 3.5, 5.0, 7.0, 10.0]
SWEEP_STEPS = 300_000       # short budget per sweep candidate
SWEEP_EVAL_EPISODES = 3

WINNING_SPEED_WEIGHT = 10.0  # <-- set this from the sweep CSV before running MODE="final"
FINAL_STEPS = 8_000_000
FINAL_SEEDS = [0, 1, 2]     # best-of-N: train this many seeds, keep the best
# ============================================================================


class CurriculumHumanoidWrapper(gym.Wrapper):
    def __init__(self, env, min_height=0.7, max_speed_cap=50.0,
                 alive_bonus=1.0, ctrl_cost_weight=0.05, smoothness_weight=0.02,
                 power_weight=0.0, cot_bonus_weight=0.0, slip_weight=0.0,
                 min_moving_speed=0.5, max_power_cost=1.0, max_slip_cost=1.0,
                 speed_weight=1.0):
        super().__init__(env)
        self.min_height = min_height
        self.max_speed_cap = max_speed_cap
        self.alive_bonus = alive_bonus
        self.ctrl_cost_weight = ctrl_cost_weight
        self.smoothness_weight = smoothness_weight
        self.power_weight = power_weight
        self.cot_bonus_weight = cot_bonus_weight
        self.slip_weight = slip_weight
        self.min_moving_speed = min_moving_speed
        self.max_power_cost = max_power_cost
        self.max_slip_cost = max_slip_cost
        self.speed_weight = speed_weight
        self._prev_action = None
        self._total_mass = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def _get_total_mass(self, model):
        if self._total_mass is None:
            self._total_mass = float(np.sum(model.body_mass))
        return self._total_mass

    def _mechanical_power(self, data):
        return float(np.sum(np.abs(data.qfrc_actuator * data.qvel)))

    def _friction_slip_loss(self, model, data):
        total_slip_power = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            try:
                force6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force6)
                tangential_force = np.linalg.norm(force6[1:3])
                if tangential_force < 1e-6:
                    continue
                geom1_body = model.geom_bodyid[contact.geom1]
                geom2_body = model.geom_bodyid[contact.geom2]
                v1 = data.cvel[geom1_body][3:6]
                v2 = data.cvel[geom2_body][3:6]
                rel_vel = v1 - v2
                contact_frame = contact.frame.reshape(3, 3)
                normal = contact_frame[0]
                rel_vel_tangential = rel_vel - np.dot(rel_vel, normal) * normal
                slip_speed = np.linalg.norm(rel_vel_tangential)
                total_slip_power += tangential_force * slip_speed
            except Exception:
                continue
        return total_slip_power

    @staticmethod
    def _soft_cap(raw_cost, cap):
        if cap <= 0:
            return raw_cost
        return cap * np.tanh(raw_cost / cap)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        obs, _env_reward, terminated, truncated, info = self.env.step(action)

        model = self.env.unwrapped.model
        data = self.env.unwrapped.data

        torso_height = data.qpos[2]
        fell = torso_height < self.min_height
        if fell:
            terminated = True

        forward_velocity = data.qvel[0]
        speed_reward = self.speed_weight * np.clip(forward_velocity, -self.max_speed_cap, self.max_speed_cap)

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        if self._prev_action is None:
            smoothness_cost = 0.0
        else:
            smoothness_cost = self.smoothness_weight * np.sum(np.square(action - self._prev_action))
        self._prev_action = action

        power = self._mechanical_power(data)
        mass = self._get_total_mass(model)

        if self.cot_bonus_weight > 0 and abs(forward_velocity) >= self.min_moving_speed:
            cot = power / (mass * GRAVITY * abs(forward_velocity))
            cot_bonus = self.cot_bonus_weight / (1.0 + cot)
        else:
            cot = float("inf")
            cot_bonus = 0.0

        power_cost = self._soft_cap(self.power_weight * power, self.max_power_cost) \
            if self.power_weight > 0 else 0.0

        if self.slip_weight > 0:
            slip_raw = self._friction_slip_loss(model, data)
            slip_cost = self._soft_cap(self.slip_weight * slip_raw, self.max_slip_cost)
        else:
            slip_cost = 0.0

        reward = (
            speed_reward
            + self.alive_bonus
            + cot_bonus
            - ctrl_cost
            - smoothness_cost
            - power_cost
            - slip_cost
        )
        if fell:
            reward -= 10.0

        if not np.isfinite(reward):
            reward = -10.0

        info = dict(info)
        info["forward_velocity"] = float(forward_velocity)
        info["mechanical_power_w"] = float(power)
        info["cost_of_transport"] = float(cot)
        info["slip_cost"] = float(slip_cost)

        return obs, float(reward), terminated, truncated, info


BASE_KWARGS = dict(
    min_height=0.7,
    alive_bonus=1.0,
    ctrl_cost_weight=0.05,
    smoothness_weight=0.02,
    power_weight=0.0002,
    cot_bonus_weight=0.0,
    slip_weight=0.0002,
    max_power_cost=3.0,
    max_slip_cost=3.0,
    max_speed_cap=50.0,
)


def make_wrapped_env(env_id, wrapper_kwargs):
    def _init():
        env = gym.make(
            env_id,
            healthy_z_range=(0.0, float("inf")),
            terminate_when_unhealthy=False,
        )
        return CurriculumHumanoidWrapper(env, **wrapper_kwargs)
    return _init


def quick_eval(model, stats_path, wrapper_kwargs, env_id="Humanoid-v5", n_episodes=3):
    """Fast, non-rendered deterministic evaluation. Returns dict of averages."""
    base_env = gym.make(
        env_id, healthy_z_range=(0.0, float("inf")), terminate_when_unhealthy=False,
    )
    wrapped_env = CurriculumHumanoidWrapper(base_env, **wrapper_kwargs)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    if os.path.exists(stats_path):
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    velocities, powers, steps_list = [], [], []
    obs = vec_env.reset()
    for _ in range(n_episodes):
        done = False
        step_count = 0
        ep_v, ep_p = [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done = bool(done_arr[0])
            step_count += 1
            ep_v.append(info[0].get("forward_velocity", 0.0))
            ep_p.append(info[0].get("mechanical_power_w", 0.0))
        velocities.append(float(np.mean(ep_v)))
        powers.append(float(np.mean(ep_p)))
        steps_list.append(step_count)
        obs = vec_env.reset()
    vec_env.close()
    return dict(
        mean_velocity=float(np.mean(velocities)),
        mean_power=float(np.mean(powers)),
        mean_steps=float(np.mean(steps_list)),
    )


def train_one_config(env_id, wrapper_kwargs, total_steps, seed, source_model_path,
                      source_vecnorm_path, save_model_path, save_vecnorm_path,
                      models_dir, tag):
    cpu_count = os.cpu_count() or 8
    n_envs = max(1, cpu_count - 2)

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = make_vec_env(make_wrapped_env(env_id, wrapper_kwargs), n_envs=n_envs,
                              vec_env_cls=vec_env_cls, seed=seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_wrapped_env(env_id, wrapper_kwargs), n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    checkpoint_dir = os.path.join(models_dir, f"checkpoints_{tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir, log_path=models_dir,
                                  eval_freq=max(40000 // n_envs, 1000), deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=max(200000 // n_envs, 1000),
                                              save_path=checkpoint_dir, name_prefix=tag, save_vecnormalize=True)
    callback = CallbackList([eval_callback, checkpoint_callback])

    policy_kwargs = dict(net_arch=dict(pi=[512, 512], qf=[512, 512]))

    if source_model_path and os.path.exists(f"{source_model_path}.zip"):
        print(f"[{tag}] Loading '{source_model_path}.zip' to continue training...")
        if source_vecnorm_path and os.path.exists(source_vecnorm_path):
            train_env = VecNormalize.load(source_vecnorm_path, train_env.venv)
            train_env.training = True
            train_env.norm_reward = True
        model = SAC.load(f"{source_model_path}.zip", env=train_env, device="cuda", seed=seed)
        model.ent_coef = "auto"
        reset_num_timesteps = False
    else:
        print(f"[{tag}] No source checkpoint found -- training fresh (seed={seed}).")
        model = SAC(
            "MlpPolicy", train_env, verbose=0, learning_rate=0.0003, buffer_size=500000,
            batch_size=256, tau=0.005, gamma=0.99, ent_coef="auto", use_sde=True,
            sde_sample_freq=4, policy_kwargs=policy_kwargs, device="cuda", seed=seed,
        )
        reset_num_timesteps = True

    print(f"[{tag}] Training for {total_steps:,} steps across {n_envs} envs (seed={seed})...")
    model.learn(total_timesteps=total_steps, callback=callback, reset_num_timesteps=reset_num_timesteps)

    train_env.save(save_vecnorm_path)
    train_env.close()
    eval_env.close()

    generic_saved_file = os.path.join(models_dir, "best_model.zip")
    if os.path.exists(generic_saved_file):
        if os.path.exists(f"{save_model_path}.zip"):
            os.remove(f"{save_model_path}.zip")
        os.rename(generic_saved_file, f"{save_model_path}.zip")

    return model


def run_sweep(models_dir):
    env_id = "Humanoid-v5"
    source_model_path = os.path.join(models_dir, "sac_humanoid_stage2")
    source_vecnorm_path = os.path.join(models_dir, "vecnormalize_stage2.pkl")

    if not os.path.exists(f"{source_model_path}.zip"):
        print(f"ERROR: expected Stage-2 checkpoint at '{source_model_path}.zip'. "
              f"Run 06_humanoid_curriculum.py (STAGE=2) first.")
        return

    results = []
    csv_path = os.path.join(models_dir, "sprinter_sweep_results.csv")

    for sw in SWEEP_SPEED_WEIGHTS:
        tag = f"sprint_sweep_sw{str(sw).replace('.', 'p')}"
        kwargs = dict(BASE_KWARGS, speed_weight=sw)
        save_model_path = os.path.join(models_dir, tag)
        save_vecnorm_path = os.path.join(models_dir, f"vecnormalize_{tag}.pkl")

        model = train_one_config(
            env_id, kwargs, SWEEP_STEPS, seed=0,
            source_model_path=source_model_path, source_vecnorm_path=source_vecnorm_path,
            save_model_path=save_model_path, save_vecnorm_path=save_vecnorm_path,
            models_dir=models_dir, tag=tag,
        )
        eval_result = quick_eval(model, save_vecnorm_path, kwargs, env_id, SWEEP_EVAL_EPISODES)
        eval_result["speed_weight"] = sw
        eval_result["survived_full_episode"] = eval_result["mean_steps"] >= 999
        results.append(eval_result)
        print(f"[sweep] speed_weight={sw} -> velocity={eval_result['mean_velocity']:.2f} m/s, "
              f"power={eval_result['mean_power']:.1f} W, "
              f"survived_full={eval_result['survived_full_episode']}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speed_weight", "mean_velocity", "mean_power",
                                                "mean_steps", "survived_full_episode"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nSweep results written to '{csv_path}'")
    stable = [r for r in results if r["survived_full_episode"]]
    pool = stable if stable else results
    best = max(pool, key=lambda r: r["mean_velocity"])
    print(f"\nBest stable config: speed_weight={best['speed_weight']} "
          f"-> {best['mean_velocity']:.2f} m/s")
    print(f"Set WINNING_SPEED_WEIGHT = {best['speed_weight']} and MODE = 'final' to continue.")


def run_final(models_dir):
    env_id = "Humanoid-v5"
    source_model_path = os.path.join(models_dir, "sac_humanoid_stage2")
    source_vecnorm_path = os.path.join(models_dir, "vecnormalize_stage2.pkl")
    kwargs = dict(BASE_KWARGS, speed_weight=WINNING_SPEED_WEIGHT)

    best_model = None
    best_score = -float("inf")
    best_seed = None

    for seed in FINAL_SEEDS:
        tag = f"sprinter_seed{seed}"
        save_model_path = os.path.join(models_dir, tag)
        save_vecnorm_path = os.path.join(models_dir, f"vecnormalize_{tag}.pkl")

        model = train_one_config(
            env_id, kwargs, FINAL_STEPS, seed=seed,
            source_model_path=source_model_path, source_vecnorm_path=source_vecnorm_path,
            save_model_path=save_model_path, save_vecnorm_path=save_vecnorm_path,
            models_dir=models_dir, tag=tag,
        )
        eval_result = quick_eval(model, save_vecnorm_path, kwargs, env_id, n_episodes=5)
        print(f"[final] seed={seed} -> velocity={eval_result['mean_velocity']:.2f} m/s, "
              f"power={eval_result['mean_power']:.1f} W")

        if eval_result["mean_velocity"] > best_score:
            best_score = eval_result["mean_velocity"]
            best_model = (save_model_path, save_vecnorm_path)
            best_seed = seed

    print(f"\nBest seed: {best_seed} -> {best_score:.2f} m/s")
    final_model_path = os.path.join(models_dir, "sac_humanoid_sprinter")
    final_vecnorm_path = os.path.join(models_dir, "vecnormalize_sprinter.pkl")
    import shutil
    shutil.copy(f"{best_model[0]}.zip", f"{final_model_path}.zip")
    shutil.copy(best_model[1], final_vecnorm_path)
    print(f"Best sprinter saved as '{final_model_path}.zip'")

    run_visual_test(final_model_path, final_vecnorm_path, kwargs, n_episodes=3)


def run_visual_test(best_model_path, stats_path, wrapper_kwargs, n_episodes=3):
    print("\nLaunching 3D window to watch the sprinter run...")
    env_id = "Humanoid-v5"
    base_env = gym.make(env_id, render_mode="human", healthy_z_range=(0.0, float("inf")),
                         terminate_when_unhealthy=False)
    try:
        base_env.unwrapped.model.geom_size[0, :] = [1000.0, 1000.0, 1.0]
    except Exception:
        pass

    wrapped_env = CurriculumHumanoidWrapper(base_env, **wrapper_kwargs)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    if os.path.exists(stats_path):
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = SAC.load(f"{best_model_path}.zip", env=vec_env, device="cuda")

    obs = vec_env.reset()
    try:
        for episode in range(n_episodes):
            print(f"\n=== Episode {episode + 1} ===")
            done = False
            step_count = 0
            velocities, powers = [], []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                try:
                    obs, reward, done_arr, info = vec_env.step(action)
                    done = bool(done_arr[0])
                except Exception:
                    return
                step_count += 1
                velocities.append(info[0].get("forward_velocity", 0.0))
                powers.append(info[0].get("mechanical_power_w", 0.0))
                time.sleep(1.0 / 60.0)
                if done:
                    print(f"Steps survived: {step_count}")
                    print(f"Avg forward velocity: {np.mean(velocities):.2f} m/s")
                    print(f"Avg mechanical power: {np.mean(powers):.1f} W")
                    time.sleep(1.5)
            obs = vec_env.reset()
    finally:
        try:
            vec_env.close()
        except Exception:
            pass


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    if MODE == "sweep":
        run_sweep(models_dir)
    elif MODE == "final":
        run_final(models_dir)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()