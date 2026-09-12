import os
import time
import warnings
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

class SymmetricGaitHumanoidWrapper(gym.Wrapper):
    """
    Forces biological symmetry by penalizing the agent if one side of the body 
    is permanently stuck forward while the other drags behind.
    """
    def __init__(self, env, min_height=0.85):
        super().__init__(env)
        self.min_height = min_height
        self.right_hip_history = []
        self.left_hip_history = []
        
    def step(self, action):
        # Clip actions to prevent extreme numerical overflow inputs to MuJoCo
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Torso Height Rule
        current_torso_height = self.env.unwrapped.data.qpos[2]
        if current_torso_height < self.min_height:
            terminated = True
            reward -= 50.0
            return obs, reward, terminated, truncated, info
            
        # 2. Extract Left/Right Hips (Indices 7 and 11 for Humanoid-v5 forward/backward swing)
        qpos = self.env.unwrapped.data.qpos
        right_hip_pitch = qpos[7]
        left_hip_pitch = qpos[11]
        
        self.right_hip_history.append(right_hip_pitch)
        self.left_hip_history.append(left_hip_pitch)
        
        if len(self.right_hip_history) > 50:
            self.right_hip_history.pop(0)
            self.left_hip_history.pop(0)
            
        # --- ASYMMETRY BREAKING PENALTY ---
        asymmetry_penalty = 0.0
        if len(self.right_hip_history) >= 30:
            right_mean = np.mean(self.right_hip_history)
            left_mean = np.mean(self.left_hip_history)
            asymmetry_penalty -= (abs(right_mean) + abs(left_mean)) * 15.0
            
        # 3. Dynamic Arm-Swing Reward
        arm_actions = action[-4:]
        arm_swing_bonus = np.sum(np.abs(arm_actions)) * 4.0
        
        stiffness_penalty = 0.0
        for joint_act in arm_actions:
            if abs(joint_act) < 0.2:
                stiffness_penalty -= 5.0
                
        reward += arm_swing_bonus + stiffness_penalty + asymmetry_penalty
        
        # Guard against NaN rewards propagating into the PyTorch optimizer
        if np.isnan(reward):
            reward = -10.0
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.right_hip_history = []
        self.left_hip_history = []
        return self.env.reset(**kwargs)

def main():
    env_id = "Humanoid-v5"
    CONTINUE_STEPS = 2000000  
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    existing_model_path = os.path.join(models_dir, "best_sac_mujoco_humanoid")
    updated_model_path = os.path.join(models_dir, "best_sac_mujoco_humanoid_natural")
    
    if not os.path.exists(f"{existing_model_path}.zip"):
        print(f"Error: Could not find '{existing_model_path}.zip'.")
        return

    print("Initializing 3D MuJoCo environment with stabilized constraints...")
    base_train_env = gym.make(env_id)
    train_env = SymmetricGaitHumanoidWrapper(base_train_env, min_height=0.85)
    
    base_eval_env = gym.make(env_id)
    eval_env = SymmetricGaitHumanoidWrapper(base_eval_env, min_height=0.85)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=models_dir, 
        eval_freq=20000,
        deterministic=True, 
        render=False
    )
    
    print("Loading pre-trained 10M step model weights onto CUDA...")
    model = SAC.load(existing_model_path, env=train_env, device="cuda")
    
    # # --- NUMERICAL STABILIZATION CONFIGURATION FIXED ---
    # # 1. Update the top-level model learning rate
    # NEW_LR = 0.0001
    # model.learning_rate = NEW_LR
    
    # # 2. Directly inject the new learning rate into PyTorch's active optimizer groups
    # for param_group in model.actor.optimizer.param_groups:
    #     param_group['lr'] = NEW_LR
    # for param_group in model.critic.optimizer.param_groups:
    #     param_group['lr'] = NEW_LR
        
    # # 3. Soft target entropy instead of a hard variable wipeout prevents variance collapse
    # model.target_entropy = -float(np.prod(train_env.action_space.shape))
    # print("-> Optimizer and target entropy safely recalibrated for stable fine-tuning.")
    # # ---------------------------------------------------
    
    # print(f"Resuming training for an additional {CONTINUE_STEPS:,} fine-tuning steps...")
    # model.learn(total_timesteps=CONTINUE_STEPS, callback=eval_callback, reset_num_timesteps=False)
    
    # train_env.close()
    # eval_env.close()
    
    # generic_saved_file = os.path.join(models_dir, "best_model.zip")
    # custom_target_file = f"{updated_model_path}.zip"
    
    # if os.path.exists(generic_saved_file):
    #     if os.path.exists(custom_target_file):
    #         os.remove(custom_target_file)
    #     os.rename(generic_saved_file, custom_target_file)
    #     print(f"Success! Symmetric gait model saved as: '{custom_target_file}'")

    print("\nLaunching 3D window to watch the trained agent play...")
    base_test_env = gym.make(env_id, render_mode="human")
    try:
        base_test_env.unwrapped.model.geom_size[0, :] = [1000.0, 1000.0, 1.0]
    except Exception:
        pass
        
    test_env = SymmetricGaitHumanoidWrapper(base_test_env, min_height=0.85)
    model = SAC.load(updated_model_path, env=test_env, device="cuda")
    
    obs, _ = test_env.reset()
    try:
        for episode in range(3):
            print(f"Starting Visual Episode {episode + 1}")
            terminated, truncated = False, False
            step_count = 0
            
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                try:
                    obs, reward, terminated, truncated, info = test_env.step(action)
                except Exception:
                    return
                step_count += 1
                time.sleep(1.0 / 60.0) 
                
                if terminated or truncated:
                    print(f"-> Episode finished at step {step_count}! Freezing screen...")
                    time.sleep(1.5)
            obs, _ = test_env.reset()
    finally:
        try: test_env.close()
        except Exception: pass

if __name__ == "__main__":
    main()
