"""
MARATHONER: fine-tunes from the Stage 2 checkpoint with efficiency
(low Cost of Transport, low mechanical power) weighted heavily over raw
speed. This agent should sacrifice velocity for genuinely low per-distance
energy cost -- caps on power/slip are tighter here, unlike the sprinter
where they're deliberately loose.
"""
import os
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
TOTAL_STEPS = 2_000_000


class CurriculumHumanoidWrapper(gym.Wrapper):
    def __init__(self, env, min_height=0.7, max_speed_cap=10.0,
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


# Marathoner profile: efficiency dominates, speed is secondary.
# NOTE: power_weight/max_power_cost recalibrated for the ~3000W power range
# actually observed after curriculum fine-tuning (earlier values were tuned
# against ~100-800W observed in prior stages and had silently saturated --
# raw_cost/cap was ~10x, putting tanh in its near-zero-gradient region,
# which gave almost no pressure to actually reduce power).
MARATHONER_KWARGS = dict(
    min_height=0.7,
    alive_bonus=1.0,
    ctrl_cost_weight=0.05,
    smoothness_weight=0.02,
    speed_weight=0.5,        # de-emphasize raw velocity
    power_weight=0.0001,     # rescaled: ~3000W raw power -> ~0.3 before cap
    cot_bonus_weight=0.15,   # doubled: primary efficiency signal, strengthened
    slip_weight=0.001,
    max_power_cost=0.3,
    max_slip_cost=0.3,
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


def main():
    env_id = "Humanoid-v5"

    cpu_count = os.cpu_count() or 8
    N_ENVS = max(1, cpu_count - 2)
    print(f"MARATHONER | Detected {cpu_count} CPU threads -> using {N_ENVS} parallel envs.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    # Fine-tune from the Stage 2 checkpoint (already has a working,
    # moderately efficient gait) rather than starting from scratch.
    source_model_path = os.path.join(models_dir, "sac_humanoid_stage2")
    source_vecnorm_path = os.path.join(models_dir, "vecnormalize_stage2.pkl")

    best_model_path = os.path.join(models_dir, "sac_humanoid_marathoner")
    vecnorm_path = os.path.join(models_dir, "vecnormalize_marathoner.pkl")
    checkpoint_dir = os.path.join(models_dir, "checkpoints_marathoner")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing vectorized MuJoCo environments (marathoner reward)...")
    vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    train_env = make_vec_env(
        make_wrapped_env(env_id, MARATHONER_KWARGS),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(
        make_wrapped_env(env_id, MARATHONER_KWARGS),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=models_dir,
        eval_freq=max(40000 // N_ENVS, 1000),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // N_ENVS, 1000),
        save_path=checkpoint_dir,
        name_prefix="sac_humanoid_marathoner",
        save_vecnormalize=True,
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    print(f"Loading checkpoint from '{source_model_path}.zip' to fine-tune toward marathon efficiency...")
    if os.path.exists(source_vecnorm_path):
        train_env = VecNormalize.load(source_vecnorm_path, train_env.venv)
        train_env.training = True
        train_env.norm_reward = True
    model = SAC.load(f"{source_model_path}.zip", env=train_env, device="cuda")
    model.ent_coef = "auto"  # re-enable exploration for fine-tuning

    print(f"Training MARATHONER for {TOTAL_STEPS:,} steps across {N_ENVS} parallel envs...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback, reset_num_timesteps=False)

    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()

    generic_saved_file = os.path.join(models_dir, "best_model.zip")
    custom_target_file = f"{best_model_path}.zip"
    if os.path.exists(generic_saved_file):
        if os.path.exists(custom_target_file):
            os.remove(custom_target_file)
        os.rename(generic_saved_file, custom_target_file)
        print(f"Saved model as: '{custom_target_file}'")

    run_visual_test(best_model_path, vecnorm_path, MARATHONER_KWARGS, n_episodes=3)


def run_visual_test(best_model_path, stats_path, wrapper_kwargs, n_episodes=3):
    print("\nLaunching 3D window to watch the marathoner run...")
    env_id = "Humanoid-v5"

    base_env = gym.make(
        env_id,
        render_mode="human",
        healthy_z_range=(0.0, float("inf")),
        terminate_when_unhealthy=False,
    )
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
    else:
        print("Warning: VecNormalize stats not found -- observations will be unnormalized.")

    model = SAC.load(f"{best_model_path}.zip", env=vec_env, device="cuda")

    all_episode_stats = []
    obs = vec_env.reset()
    try:
        for episode in range(n_episodes):
            print(f"\n=== Episode {episode + 1} ===")
            done = False
            step_count = 0
            velocities, powers, cots, slips = [], [], [], []

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                try:
                    obs, reward, done_arr, info = vec_env.step(action)
                    done = bool(done_arr[0])
                except Exception:
                    return
                step_count += 1

                velocities.append(info[0].get("forward_velocity", 0.0))
                powers.append(info[0].get("mechanical_power_w", 0.0))
                cots.append(info[0].get("cost_of_transport", 0.0))
                slips.append(info[0].get("slip_cost", 0.0))

                time.sleep(1.0 / 60.0)

                if done:
                    avg_v = float(np.mean(velocities))
                    avg_power = float(np.mean(powers))
                    finite_cots = [c for c in cots if np.isfinite(c)]
                    avg_cot = float(np.mean(finite_cots)) if finite_cots else float("inf")
                    pct_moving = 100.0 * len(finite_cots) / max(len(cots), 1)
                    avg_slip = float(np.mean(slips))
                    print(f"Steps survived: {step_count}")
                    print(f"Avg forward velocity: {avg_v:.2f} m/s")
                    print(f"Avg mechanical power: {avg_power:.1f} W")
                    print(f"Avg cost of transport (while moving, {pct_moving:.0f}% of steps): {avg_cot:.3f}")
                    print(f"Avg slip-loss term: {avg_slip:.4f}")
                    all_episode_stats.append((step_count, avg_v, avg_power, avg_cot, avg_slip))
                    time.sleep(1.5)

            obs = vec_env.reset()
    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    if all_episode_stats:
        steps, vs, pws, cots, slips = zip(*all_episode_stats)
        print("\n=== Summary across all episodes ===")
        print(f"Mean steps survived: {np.mean(steps):.0f}")
        print(f"Mean velocity:       {np.mean(vs):.2f} m/s")
        print(f"Mean power:          {np.mean(pws):.1f} W")
        print(f"Mean cost of transport: {np.mean(cots):.3f}")
        print(f"Mean slip-loss:      {np.mean(slips):.4f}")


if __name__ == "__main__":
    main()