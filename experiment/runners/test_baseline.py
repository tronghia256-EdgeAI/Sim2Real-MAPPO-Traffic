from __future__ import annotations

import time
from pathlib import Path

from src.traffic_env.config import build_default_config
from src.traffic_env.envs.multi_agent import MappoTrafficEnv

# ========================= CONFIG =========================
# Path to testing SUMO config
import sys
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SUMO_CFG_PATH = BASE_DIR / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"

# Fixed-time logic: Green(60 steps) -> Yellow(3 steps)
FIXED_PHASES = [
    (0, 60),  # Phase 0: Green NS (60 steps)
    (1, 3),   # Phase 1: Yellow NS (3 steps)
    (2, 60),  # Phase 2: Green EW (60 steps)
    (3, 3)    # Phase 3: Yellow EW (3 steps)
]

def main():
    print("\n" + "="*40)
    print("🚦 STARTING FIXED-TIME BASELINE (NO PPO)")
    print("="*40)

    # Check config file
    if not SUMO_CFG_PATH.exists():
        print(f"❌ ERROR: SUMO config not found at:\n{SUMO_CFG_PATH}")
        return

    print("🚥 Initializing SUMO Environment (GUI Mode)...")
    cfg = build_default_config(sumo_cfg_path=str(SUMO_CFG_PATH), gui=True)
    env = MappoTrafficEnv(config=cfg, gui=True, use_libsumo=False)
    env.reset()
    
    done = False
    total_reward = 0.0
    step_count = 0
    
    current_phase_idx = 0
    timer = 0

    print("\n🔴 SIMULATION STARTED (FIXED-TIME MODE)...")
    try:
        while not done:
            # Get current phase and duration
            action_val, duration = FIXED_PHASES[current_phase_idx]
            
            # Apply same action to both intersections (J2, J0)
            action_dict = {tls_id: int(action_val in (2, 3)) for tls_id in cfg.tls_ids}
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_dict)
            done = any(terminated.values()) or any(truncated.values())
            total_reward += float(sum(reward.values())) if reward else 0.0
            step_count += 1
            timer += 1

            # Log every 50 steps
            if step_count % 50 == 0:
                print(f"Step: {step_count:<5} | Reward: {total_reward:7.2f} | Phase: {action_val}")

            # Switch to next phase when duration ends
            if timer >= duration:
                timer = 0
                current_phase_idx = (current_phase_idx + 1) % len(FIXED_PHASES)

            # Small delay for GUI observation
            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
    finally:
        print("\n" + "="*40)
        print("🏁 FIXED-TIME BASELINE DONE")
        print(f"📊 Total Steps: {step_count}")
        print(f"💰 Total Reward (Penalty): {total_reward:.2f}")
        print("="*40)
        env.close()

if __name__ == "__main__":
    main()