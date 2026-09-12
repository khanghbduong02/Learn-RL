import os
import time
import warnings
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# Suppress the Stable-Baselines3 GPU policy warning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# 1. CREATE A POSTURE WRAPPER TO FORBID CROUCH-WALKING
class TallHumanoidWrapper(gym.Wrapper):
    """
    Terminates the episode instantly if the humanoid drops its hips or slouches,
    forcing it to learn to walk tall and leverage its full leg extension.
    """
    def __init__(self, env, min_height=0.85):
        super().__init__(env)
        self.min_height = min_height
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # In Humanoid-v5, the root torso coordinate height is tracked.
        # Alternatively, we can grab the exact current z-position directly from MuJoCo memory layout
        current_torso_height = self.env.unwrapped.data.qpos[2]
        
        # If the humanoid crouches or falls below our threshold, end the trial instantly
        if current_torso_height < self.min_height:
            terminated = True
            reward -= 50.0  # Heavy penalty for collapsing or slouching
            
        return obs, reward, terminated, truncated, info

def main():
    # 2. Configuration Setup
    env_id = "Humanoid-v5"
    TOTAL_STEPS = 10000000  # Scaled to 10 Million steps for complex gait optimization
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    best_model_path = os.path.join(models_dir, "best_sac_mujoco_humanoid")
    
    # 3. Train the Model on CUDA (GPU)
    print(f"Initializing 3D MuJoCo {env_id} environment with Tall Posture Wrapper...")
    base_train_env = gym.make(env_id)
    # Apply our custom tall walking rule (Default standing height is ~1.25m, so 0.85m keeps it upright)
    train_env = TallHumanoidWrapper(base_train_env, min_height=0.85)
    
    base_eval_env = gym.make(env_id)
    eval_env = TallHumanoidWrapper(base_eval_env, min_height=0.85)
    
    # Checkpoint evaluation system to capture the peak posture configurations
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=models_dir, 
        eval_freq=50000,  # Checked every 50k steps to manage the massive 10M run
        deterministic=True, 
        render=False
    )
    
    print("Setting up stabilized SAC algorithm on CUDA...")
    # Expanded deep layer widths to manage 17 continuous joint calculations simultaneously
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    
    model = SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=0.0003, 
        buffer_size=500000, # Increased buffer slightly to retain diverse motion patterns
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        device="cuda"
    )
    
    print(f"Training the tall humanoid on GPU for {TOTAL_STEPS:,} steps. Please wait...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    
    train_env.close()
    eval_env.close()
    
    # Automatically handle filename organization
    generic_saved_file = os.path.join(models_dir, "best_model.zip")
    custom_target_file = f"{best_model_path}.zip"
    
    if os.path.exists(generic_saved_file):
        if os.path.exists(custom_target_file):
            os.remove(custom_target_file)
        os.rename(generic_saved_file, custom_target_file)
        print(f"Success! Best model renamed and saved as: '{custom_target_file}'")
    else:
        print("Warning: Peak model file not found.")

    # 4. Watch the Trained Tall Humanoid Walk
    print("\nLaunching 3D window to watch the trained agent play...")
    base_test_env = gym.make(env_id, render_mode="human")
    
    # Expand ground floor texture area so it can walk indefinitely
    try:
        base_test_env.unwrapped.model.geom_size[0, :] = [1000.0, 1000.0, 1.0]
        print("-> Checkered ground floor area expanded successfully to a 1000x1000 zone!")
    except Exception as e:
        print(f"Note: Could not automatically upscale the floor texture grid: {e}")
        
    test_env = TallHumanoidWrapper(base_test_env, min_height=0.85)
    model = SAC.load(best_model_path, env=test_env, device="cuda")
    
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
                    print("Window manually closed or ESC pressed. Exiting cleanly...")
                    return
                    
                step_count += 1
                time.sleep(1.0 / 60.0) 
                
                if terminated or truncated:
                    if terminated:
                        print(f"-> Humanoid lost posture balance or collapsed at step {step_count}! Freezing...")
                    elif truncated:
                        print(f"-> Reached maximum 1000 step limit! Beautiful tall walk! Freezing...")
                    time.sleep(1.5)
                
            obs, _ = test_env.reset()
    finally:
        try:
            test_env.close()
        except Exception:
            pass
            
    print("Simulation finished successfully!")

if __name__ == "__main__":
    main()
