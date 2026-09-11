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

# ============================================================================
# STAGE SELECTOR
#   STAGE = 1: proven speed-only reward (matches the earlier run that
#              successfully sustained 1000 steps at 3+ m/s). No power/CoT/
#              friction terms -- those default to weight 0 below. Use this
#              to (re)build a stable base gait from scratch.
#   STAGE = 2: loads the Stage-1 checkpoint and CONTINUES training with
#              small physics-efficiency weights turned on, so the agent
#              refines an already-working gait instead of learning
#              locomotion and efficiency simultaneously from scratch.
# Run Stage 1 to completion first, confirm it still runs well via
# run_visual_test, THEN switch this to 2 and rerun.
# ============================================================================
STAGE = 1

TOTAL_STEPS_STAGE1 = 5_000_000
TOTAL_STEPS_STAGE2 = 2_000_000  # fine-tuning needs far fewer steps


class CurriculumHumanoidWrapper(gym.Wrapper):
    """
    Unified reward wrapper used for both curriculum stages, so the
    observation/action space and base reward structure never change
    between stages -- only the efficiency-term weights differ. This lets
    Stage 2 load the Stage-1 policy and fine-tune rather than relearn.

    Base terms (always active, proven to produce sustained locomotion):
      - speed_reward: bounded forward velocity.
      - alive_bonus: flat per-step bonus.
      - ctrl_cost: action-magnitude penalty (sum of squared actions),
        NOT physical power -- this is the same cheap regularizer from the
        version that worked.
      - smoothness_cost: frame-to-frame action jerk penalty.

    Efficiency terms (weight 0 in Stage 1, small nonzero in Stage 2):
      - power_cost: real mechanical power (qfrc_actuator * qvel), Watts.
      - cot_bonus: Cost-of-Transport based efficiency bonus, only applied
        above min_moving_speed (never rewards standing still).
      - slip_cost: approximate friction-slip energy loss.

    All costs use a SOFT cap (tanh saturation) rather than a hard min()
    clip. A hard clip creates a flat zero-marginal-cost region once the
    cap is hit -- which is exactly what caused the "flail with 900W of
    torque because extra power is free past the cap" failure. tanh
    saturation still asymptotically bounds the penalty but never fully
    flattens the marginal incentive to reduce it further.
    """

    def __init__(self, env, min_height=0.7, max_speed_cap=10.0,
                 alive_bonus=1.0, ctrl_cost_weight=0.05, smoothness_weight=0.02,
                 power_weight=0.0, cot_bonus_weight=0.0, slip_weight=0.0,
                 min_moving_speed=0.5, max_power_cost=1.0, max_slip_cost=1.0):
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
        # Smooth saturation: bounded like a hard cap, but marginal cost
        # never fully vanishes (derivative of tanh is never exactly 0),
        # so there's no "extra effort is free past this point" region.
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
        speed_reward = np.clip(forward_velocity, -self.max_speed_cap, self.max_speed_cap)

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        if self._prev_action is None:
            smoothness_cost = 0.0
        else:
            smoothness_cost = self.smoothness_weight * np.sum(np.square(action - self._prev_action))
        self._prev_action = action

        # --- Efficiency terms (inert at weight 0, i.e. Stage 1) ---
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


def make_wrapped_env(env_id, wrapper_kwargs):
    def _init():
        env = gym.make(
            env_id,
            healthy_z_range=(0.0, float("inf")),
            terminate_when_unhealthy=False,
        )
        return CurriculumHumanoidWrapper(env, **wrapper_kwargs)
    return _init


def get_stage_kwargs(stage):
    if stage == 1:
        # Exactly the config that previously produced a sustained
        # ~3+ m/s gait for the full 1000-step episode. Efficiency
        # terms OFF.
        return dict(
            min_height=0.7,
            alive_bonus=1.0,
            ctrl_cost_weight=0.05,
            smoothness_weight=0.02,
            power_weight=0.0,
            cot_bonus_weight=0.0,
            slip_weight=0.0,
        )
    elif stage == 2:
        # Same base config, efficiency terms turned on at SMALL weights.
        # These are still guesses -- validate on a short run before
        # trusting them, same as any new weight.
        return dict(
            min_height=0.7,
            alive_bonus=1.0,
            ctrl_cost_weight=0.05,
            smoothness_weight=0.02,
            power_weight=0.0005,
            cot_bonus_weight=0.02,
            slip_weight=0.0005,
            max_power_cost=0.5,
            max_slip_cost=0.5,
        )
    else:
        raise ValueError(f"Unknown STAGE: {stage}")


def main():
    env_id = "Humanoid-v5"
    stage_kwargs = get_stage_kwargs(STAGE)
    total_steps = TOTAL_STEPS_STAGE1 if STAGE == 1 else TOTAL_STEPS_STAGE2

    cpu_count = os.cpu_count() or 8
    N_ENVS = max(1, cpu_count - 2)
    print(f"STAGE {STAGE} | Detected {cpu_count} CPU threads -> using {N_ENVS} parallel envs.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    stage1_model_path = os.path.join(models_dir, "sac_humanoid_stage1")
    stage1_vecnorm_path = os.path.join(models_dir, "vecnormalize_stage1.pkl")
    stage2_model_path = os.path.join(models_dir, "sac_humanoid_stage2")
    stage2_vecnorm_path = os.path.join(models_dir, "vecnormalize_stage2.pkl")

    best_model_path = stage1_model_path if STAGE == 1 else stage2_model_path
    vecnorm_path = stage1_vecnorm_path if STAGE == 1 else stage2_vecnorm_path

    checkpoint_dir = os.path.join(models_dir, f"checkpoints_stage{STAGE}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Initializing vectorized MuJoCo environments (Stage {STAGE} reward)...")
    vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    train_env = make_vec_env(
        make_wrapped_env(env_id, stage_kwargs),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(
        make_wrapped_env(env_id, stage_kwargs),
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
        name_prefix=f"sac_humanoid_stage{STAGE}",
        save_vecnormalize=True,
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    policy_kwargs = dict(net_arch=dict(pi=[512, 512], qf=[512, 512]))

    if STAGE == 1:
        print("Setting up fresh SAC model (Stage 1: speed-only) on CUDA...")
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
    else:
        print(f"Loading Stage-1 checkpoint from '{stage1_model_path}.zip' to continue training "
              f"with efficiency terms enabled...")
        if os.path.exists(stage1_vecnorm_path):
            train_env = VecNormalize.load(stage1_vecnorm_path, train_env.venv)
            train_env.training = True
            train_env.norm_reward = True
        model = SAC.load(f"{stage1_model_path}.zip", env=train_env, device="cuda")
        # Re-enable exploration for fine-tuning -- Stage 1's entropy will
        # have annealed down near zero by the end of training, which
        # would otherwise prevent the policy from adapting to the new
        # reward terms at all.
        model.ent_coef = "auto"

    print(f"Training Stage {STAGE} for {total_steps:,} steps across {N_ENVS} parallel envs...")
    model.learn(total_timesteps=total_steps, callback=callback, reset_num_timesteps=(STAGE == 1))

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

    run_visual_test(best_model_path, vecnorm_path, stage_kwargs, n_episodes=3)


def run_visual_test(best_model_path, stats_path, wrapper_kwargs, n_episodes=3):
    print("\nLaunching 3D window to watch the trained agent run...")
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