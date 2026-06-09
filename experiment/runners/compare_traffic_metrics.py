from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. PATH RESOLUTION
# ==========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.traffic_env.config import build_default_config
from src.traffic_env.envs.multi_agent import MappoTrafficEnv
from src.core.policy_loader import PolicyLoader

# ========================= CONFIG =========================
# Use BASE_DIR for absolute paths
MODEL_PATH = BASE_DIR / "models" / "mappo" / "20260418_215140" / "best_model.pt"
SUMO_CFG_PATH = BASE_DIR / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg" 

GUI = False  # Set False for faster data collection
MAX_STEPS = 1080  # 1 episode

# Baseline phase configuration
FIXED_PHASES = [
    (0, 60),  # Phase 0 (Green NS): 30s
    (1, 3),   # Phase 1 (Yellow NS): 3s
    (2, 60),  # Phase 2 (Green EW): 30s
    (3, 3)    # Phase 3 (Yellow EW): 3s
]

def run_scenario(scenario_name='ppo'):
    print(f"\n>>> STARTING SCENARIO: {scenario_name.upper()} <<<")
    
    # Initialize environment
    cfg = build_default_config(sumo_cfg_path=str(SUMO_CFG_PATH), gui=GUI)
    env = MappoTrafficEnv(config=cfg, gui=GUI, use_libsumo=not GUI)
    obs, _ = env.reset()
    
    # Load model if PPO
    policy = None
    if scenario_name == 'ppo':
        policy = PolicyLoader(
            checkpoint_path=MODEL_PATH,
            obs_dim=cfg.local_obs_dim,
            action_dim=2,
        ).load()
    
    # Time-series data storage
    step_history = []
    halted_history = []
    waiting_history = []
    speed_history = []
    
    # Baseline variables
    current_phase_idx = 0
    fixed_timer = 0
    
    for step in range(MAX_STEPS):
        # 1. Get action
        if scenario_name == 'ppo':
            action = policy.predict(
                {tls_id: np.asarray(obs[tls_id], dtype=np.float32) for tls_id in cfg.tls_ids},
                deterministic=True,
            )
        else:
            # Baseline timer-based action
            action_val, duration = FIXED_PHASES[current_phase_idx]
            action = {tls_id: int(action_val in (2, 3)) for tls_id in cfg.tls_ids}
            
            fixed_timer += 1
            if fixed_timer >= duration:
                fixed_timer = 0
                current_phase_idx = (current_phase_idx + 1) % len(FIXED_PHASES)

        # 2. Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = any(terminated.values()) or any(truncated.values())
        
        # --- 3. COLLECT TRAFFIC METRICS ---
        step_history.append(step * env.step_length)  # Simulation seconds
        # Sum halted vehicles across all TLS (correct key: queue_total_proxy)
        halted_total = sum(
            float(info.get(tid, {}).get('queue_total_proxy', 0.0))
            for tid in cfg.tls_ids
        )
        halted_history.append(halted_total)
        info0 = next(iter(info.values()), {})
        waiting_history.append(float(info0.get('avg_waiting_proxy', 0.0)))
        # Speed from lane cache (avg_speed_proxy does not exist in info_dict)
        lane_cache_step = env.obs_builder.build_lane_cache(env.tls_ids)
        speeds = [float(m.get('raw_speed', 0.0)) for m in lane_cache_step.values() if m]
        speed_history.append(float(np.mean(speeds)) if speeds else 0.0)

        if done or truncated: 
            break

    env.close()
    
    # Aggregate results
    results = {
        'time': step_history,
        'halted': halted_history,
        'waiting': waiting_history,
        'speed': speed_history,
        'total_waiting_time': np.sum(waiting_history),
        'mean_halted_vehicles': np.mean(halted_history),
        'mean_speed_kmh': np.mean(speed_history) * 3.6  # Convert to km/h
    }
    
    print(f">>> {scenario_name.upper()} DONE <<<")
    return results

# ========================= MAIN =========================
if __name__ == "__main__":
    # Run PPO
    ppo_results = run_scenario(scenario_name='ppo')
    
    # Run Baseline (wait 2s for SUMO ports to close properly)
    time.sleep(2)
    baseline_results = run_scenario(scenario_name='baseline')
    
    print("\n" + "="*40)
    print("=== FINAL TRAFFIC METRICS COMPARISON ===")
    print("="*40)
    
    print(f"{'Metric':<30} | {'No PPO (Fixed)':<15} | {'PPO (AI)':<15}")
    print("-" * 66)
    
    # 1. Compare Total Waiting Time
    print(f"{'Total Waiting Time (vehicle-s)':<30} | "
          f"{baseline_results['total_waiting_time']:,.1f} | "
          f"{ppo_results['total_waiting_time']:,.1f}")
    
    # 2. Compare Mean Halted Vehicles
    print(f"{'Mean Halted Vehicles (avg queue)':<30} | "
          f"{baseline_results['mean_halted_vehicles']:.2f} | "
          f"{ppo_results['mean_halted_vehicles']:.2f}")

    # 3. Compare Average System Speed
    print(f"{'Average System Speed (km/h)':<30} | "
          f"{baseline_results['mean_speed_kmh']:.2f} | "
          f"{ppo_results['mean_speed_kmh']:.2f}")
    
    print("="*40)
    
    # Calculate improvement %
    if baseline_results['total_waiting_time'] > 0:
        waiting_improvement = ((baseline_results['total_waiting_time'] - ppo_results['total_waiting_time']) / baseline_results['total_waiting_time']) * 100
        print(f"👉 AI reduced total waiting time by {waiting_improvement:.1f}%.")

    # ==========================================================
    # VISUALIZATION
    # ==========================================================
    print("\n=== GENERATING COMPARISON CHARTS ===")
    
    # Setup plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Get min length to avoid size mismatch
    min_len = min(len(baseline_results['time']), len(ppo_results['time']))
    
    # 1. Total Waiting Time Plot
    axs[0].plot(baseline_results['time'][:min_len], baseline_results['waiting'][:min_len], color='red', alpha=0.6, label='No PPO (Fixed-Time)')
    axs[0].plot(ppo_results['time'][:min_len], ppo_results['waiting'][:min_len], color='green', alpha=0.8, linewidth=2, label='PPO (AI Adaptive)')
    axs[0].set_ylabel("Total Waiting Time (seconds)")
    axs[0].set_title("1. Network Delay Comparison")
    axs[0].legend()
    
    # 2. Halted Vehicles Plot
    axs[1].plot(baseline_results['time'][:min_len], baseline_results['halted'][:min_len], color='red', alpha=0.6, label='No PPO')
    axs[1].plot(ppo_results['time'][:min_len], ppo_results['halted'][:min_len], color='green', alpha=0.8, linewidth=2, label='PPO')
    axs[1].set_ylabel("Number of Halted Vehicles")
    axs[1].set_title("2. Traffic Congestion Comparison (Queue Length)")
    axs[1].legend()

    # 3. System Speed Plot
    axs[2].plot(baseline_results['time'][:min_len], baseline_results['speed'][:min_len], color='red', alpha=0.6, label='No PPO')
    axs[2].plot(ppo_results['time'][:min_len], ppo_results['speed'][:min_len], color='green', alpha=0.8, linewidth=2, label='PPO')
    axs[2].set_xlabel("Simulation Time (seconds)")
    axs[2].set_ylabel("Average System Speed (m/s)")
    axs[2].set_title("3. Traffic Flow Comparison (System Throughput)")
    axs[2].legend()
    
    # Save chart to file
    chart_path = BASE_DIR / "results" / "graphs" / "traffic_comparison_charts.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path)
    print(f"Success! Comparison charts saved to: {chart_path}")
    
    # Show chart
    plt.show()