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


class PhysicallyGroundedHumanoidWrapper(gym.Wrapper):
    """
    Speed-focused wrapper grounded in actual simulated physics rather than
    hand-crafted gait shaping. No priors about "correct" posture or limb
    coordination — the agent is free to discover whatever locomotion
    strategy is fastest AND cheapest in real physical terms:

      - forward velocity: the primary objective (bounded, not exponential).
      - mechanical power (Watts): Power = sum(actuator_force * joint_vel)
        across all DOFs, taken directly from MuJoCo's qfrc_actuator/qvel.
        This is real instantaneous mechanical power, not a proxy.
      - Cost of Transport (CoT) = Power / (mass * g * velocity), the
        standard dimensionless efficiency metric used in real bipedal
        robotics/biomechanics to compare gaits independent of body size
        or speed. Lower is more efficient. We convert this into a reward
        bonus for efficient locomotion.
      - friction-slip energy loss: an APPROXIMATION of energy dissipated
        to kinetic friction when a contacting body slides rather than
        grips (foot dragging, knee-skating, etc). Computed from MuJoCo's
        per-contact force decomposition x an estimated relative sliding
        velocity at the contact site. This is deliberately conservative;
        verify against your MuJoCo version before trusting the exact
        magnitude — the goal is "discourage energy-wasting sliding
        contact," not a lab-grade tribology model.

    None of these terms encode what a "correct" gait looks like — they
    encode what is/isn't physically expensive, which is exactly the
    distinction you asked for.
    """

    def __init__(self, env, min_height=0.5, max_speed_cap=10.0,
                 power_weight=0.01, cot_bonus_weight=0.05,
                 slip_weight=0.02, smoothness_weight=0.02,
                 alive_bonus=1.0):
        super().__init__(env)
        self.min_height = min_height
        self.max_speed_cap = max_speed_cap
        self.power_weight = power_weight
        self.cot_bonus_weight = cot_bonus_weight
        self.slip_weight = slip_weight
        self.smoothness_weight = smoothness_weight
        self.alive_bonus = alive_bonus
        self._prev_action = None
        self._total_mass = None  # cached on first step

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def _get_total_mass(self, model):
        if self._total_mass is None:
            self._total_mass = float(np.sum(model.body_mass))
        return self._total_mass

    def _mechanical_power(self, data):
        # Real instantaneous mechanical power delivered by actuators:
        # sum over all DOFs of (generalized actuator force * joint velocity).
        # qfrc_actuator has shape (nv,), same as qvel.
        return float(np.sum(np.abs(data.qfrc_actuator * data.qvel)))

    def _friction_slip_loss(self, model, data):
        """
        Approximate energy dissipated to sliding friction across all active
        contacts this step. For each contact: tangential (friction) force
        magnitude x approximate relative sliding speed at that contact.
        Uses geom-frame velocity as a stand-in for exact contact-point
        velocity (reasonable for small/capsule contact geoms like feet,
        less exact for large flat contacts).
        """
        total_slip_power = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            try:
                force6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force6)
                tangential_force = np.linalg.norm(force6[1:3])
                if tangential_force < 1e-6:
                    continue

                # Approximate sliding speed via the two contacting geoms'
                # body linear velocities projected onto the contact frame's
                # tangent plane. This is an approximation, not an exact
                # contact-point velocity.
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
                # Defensive: contact force API details can vary by MuJoCo
                # version. Skip this contact rather than crash training.
                continue
        return total_slip_power

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

        # --- Mechanical power & Cost of Transport ---
        power = self._mechanical_power(data)
        mass = self._get_total_mass(model)
        # Guard against division blowup at near-zero velocity.
        safe_speed = max(abs(forward_velocity), 0.3)
        cot = power / (mass * GRAVITY * safe_speed)
        # Reward LOW cost of transport (efficient locomotion), scaled down
        # since CoT can be a large number early in training when movement
        # is erratic and inefficient.
        cot_bonus = self.cot_bonus_weight / (1.0 + cot)

        power_cost = self.power_weight * power

        # --- Friction / slip energy loss ---
        slip_cost = self.slip_weight * self._friction_slip_loss(model, data)

        # --- Smoothness (guards against actuator-buzzing exploits) ---
        if self._prev_action is None:
            smoothness_cost = 0.0
        else:
            smoothness_cost = self.smoothness_weight * np.sum(np.square(action - self._prev_action))
        self._prev_action = action

        reward = (
            speed_reward
            + self.alive_bonus
            + cot_bonus
            - power_cost
            - slip_cost
            - smoothness_cost
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


def make_wrapped_env(env_id, min_height=0.5):
    def _init():
        env = gym.make(
            env_id,
            healthy_z_range=(0.0, float("inf")),
            terminate_when_unhealthy=False,
        )
        return PhysicallyGroundedHumanoidWrapper(env, min_height=min_height)
    return _init


def main():
    env_id = "Humanoid-v5"
    TOTAL_STEPS = 5_000_000

    cpu_count = os.cpu_count() or 8
    N_ENVS = max(1, cpu_count - 2)
    print(f"Detected {cpu_count} CPU threads -> using {N_ENVS} parallel envs.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "best_sac_mujoco_humanoid_physical")
    vecnorm_path = os.path.join(models_dir, "vecnormalize.pkl")
    checkpoint_dir = os.path.join(models_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing vectorized MuJoCo environments (physics-grounded reward)...")

    vec_env_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    train_env = make_vec_env(
        make_wrapped_env(env_id, min_height=0.5),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(
        make_wrapped_env(env_id, min_height=0.5),
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
        name_prefix="sac_humanoid_physical",
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

    print(f"Training for {TOTAL_STEPS:,} steps across {N_ENVS} parallel envs...")
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


if __name__ == "__main__":
    main()