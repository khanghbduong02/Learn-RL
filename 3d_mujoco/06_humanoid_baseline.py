import os
import time
import warnings
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")


class RawSpeedHumanoidWrapper(gym.Wrapper):
    """
    Minimal-bias wrapper: no hand-crafted gait shaping (no stride symmetry,
    no arm-swing terms, no anatomical assumptions about "correct" running
    form). The agent is rewarded almost purely for forward velocity and left
    free to discover whatever limb coordination pattern maximizes speed.

    Non-speed terms are physical/optimization safeguards, not aesthetic ones:
      - a small alive bonus;
      - a light control cost (whole-body, all actuators treated equally);
      - a light action-smoothness cost, penalizing frame-to-frame action
        jerk. This discourages a specific SAC failure mode (rapidly
        vibrating actuators to satisfy a magnitude-only control cost while
        producing physically nonsensical motion) — it does not push the
        gait toward looking human, it just discourages a known exploit;
      - a bounded (not exponential) velocity term.
    """

    def __init__(self, env, min_height=0.7, max_speed_cap=10.0,
                 ctrl_cost_weight=0.05, smoothness_weight=0.02,
                 alive_bonus=1.0):
        super().__init__(env)
        self.min_height = min_height
        self.max_speed_cap = max_speed_cap
        self.ctrl_cost_weight = ctrl_cost_weight
        self.smoothness_weight = smoothness_weight
        self.alive_bonus = alive_bonus
        self._prev_action = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        obs, _env_reward, terminated, truncated, info = self.env.step(action)

        torso_height = self.env.unwrapped.data.qpos[2]
        fell = torso_height < self.min_height
        if fell:
            terminated = True

        forward_velocity = self.env.unwrapped.data.qvel[0]
        speed_reward = np.clip(forward_velocity, -self.max_speed_cap, self.max_speed_cap)

        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        if self._prev_action is None:
            smoothness_cost = 0.0
        else:
            smoothness_cost = self.smoothness_weight * np.sum(np.square(action - self._prev_action))
        self._prev_action = action

        reward = speed_reward + self.alive_bonus - control_cost - smoothness_cost
        if fell:
            reward -= 10.0

        if not np.isfinite(reward):
            reward = -10.0

        info = dict(info)
        info["forward_velocity"] = float(forward_velocity)

        return obs, float(reward), terminated, truncated, info


def make_wrapped_env(env_id, min_height=0.7):
    def _init():
        env = gym.make(
            env_id,
            healthy_z_range=(0.0, float("inf")),
            terminate_when_unhealthy=False,
        )
        return RawSpeedHumanoidWrapper(env, min_height=min_height)
    return _init


def main():
    env_id = "Humanoid-v5"
    TOTAL_STEPS = 5_000_000

    # Leave 1-2 cores free for the OS/logging/eval overhead. Adjust N_ENVS
    # down if you notice fps dropping due to thermal throttling on laptop.
    cpu_count = os.cpu_count() or 8
    N_ENVS = max(1, cpu_count - 2)
    print(f"Detected {cpu_count} CPU threads -> using {N_ENVS} parallel envs.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "best_sac_mujoco_humanoid_sprinter")
    vecnorm_path = os.path.join(models_dir, "vecnormalize.pkl")
    checkpoint_dir = os.path.join(models_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing vectorized MuJoCo environments (raw-speed objective, no gait priors)...")

    vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    train_env = make_vec_env(
        make_wrapped_env(env_id, min_height=0.7),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(
        make_wrapped_env(env_id, min_height=0.7),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    # Eval env shares observation normalization stats with training, but
    # does NOT normalize reward (so eval scores stay human-interpretable)
    # and does NOT update running stats (training=False).
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
        name_prefix="sac_humanoid",
        save_vecnormalize=True,
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    print("Setting up SAC on CUDA...")
    policy_kwargs = dict(net_arch=dict(pi=[512, 512], qf=[512, 512]))

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=500000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        device="cuda",
    )

    print(f"Training for {TOTAL_STEPS:,} steps across {N_ENVS} parallel envs "
          f"(watch the 'fps' column in the log to gauge real throughput)...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

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

    # Prefer the VecNormalize stats saved alongside the best eval checkpoint;
    # fall back to the final training stats saved above.
    eval_vecnorm_path = os.path.join(models_dir, "vecnormalize.pkl")
    stats_path = eval_vecnorm_path if os.path.exists(eval_vecnorm_path) else vecnorm_path

    print("\nLaunching 3D window to watch the trained agent run...")
    base_test_env = gym.make(
        env_id,
        render_mode="human",
        healthy_z_range=(0.0, float("inf")),
        terminate_when_unhealthy=False,
    )
    try:
        base_test_env.unwrapped.model.geom_size[0, :] = [1000.0, 1000.0, 1.0]
    except Exception:
        pass

    test_env_raw = RawSpeedHumanoidWrapper(base_test_env, min_height=0.7)
    test_vec_env = DummyVecEnv([lambda: test_env_raw])
    if os.path.exists(stats_path):
        test_vec_env = VecNormalize.load(stats_path, test_vec_env)
        test_vec_env.training = False
        test_vec_env.norm_reward = False
    else:
        print("Warning: VecNormalize stats not found, running with raw observations.")

    model = SAC.load(best_model_path, env=test_vec_env, device="cuda")

    obs = test_vec_env.reset()
    try:
        for episode in range(3):
            print(f"Starting Visual Episode {episode + 1}")
            done = False
            step_count = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                try:
                    obs, reward, done_arr, info = test_vec_env.step(action)
                    done = bool(done_arr[0])
                except Exception:
                    return
                step_count += 1
                time.sleep(1.0 / 60.0)

                if done:
                    v = info[0].get("forward_velocity", 0.0)
                    print(f"-> Episode finished at step {step_count} (last v={v:.2f} m/s)")
                    time.sleep(1.5)
            obs = test_vec_env.reset()
    finally:
        try:
            test_vec_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()