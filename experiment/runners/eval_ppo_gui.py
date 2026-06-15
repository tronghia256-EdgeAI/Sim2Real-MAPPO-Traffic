from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so train_ppo and src are importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.traffic_env.config import build_default_config
from src.traffic_env.envs.multi_agent import MappoTrafficEnv
from src.core.policy_loader import PolicyLoader

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "mappo" / "20260418_215140" / "best_model.pt"
SUMO_CFG_PATH = BASE_DIR / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"

def main():
    print("\n" + "="*40)
    print("🚀 SMART TRAFFIC SYSTEM - PPO EVALUATION")
    print("="*40)

    # Check config file
    if not SUMO_CFG_PATH.exists():
        print(f"❌ ERROR: SUMO config not found at:\n{SUMO_CFG_PATH}")
        return

    print("🚥 Initializing SUMO Environment (GUI Mode)...")
    cfg = build_default_config(sumo_cfg_path=str(SUMO_CFG_PATH), gui=True)

    print(f"📦 Loading Model: {MODEL_PATH.name}")
    try:
        policy = PolicyLoader(
            checkpoint_path=MODEL_PATH,
            obs_dim=cfg.local_obs_dim,
            action_dim=2,
        ).load()
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    env = MappoTrafficEnv(config=cfg, gui=True, use_libsumo=False)
    obs, _ = env.reset()

    done = False
    total_reward = 0.0
    step_count = 0
    phase_switch_counts = {tls_id: 0 for tls_id in cfg.tls_ids}
    prev_actions = {tls_id: -1 for tls_id in cfg.tls_ids}

    print("\n🟢 SIMULATION STARTED...")
    print(f"{'Step':<6} {'Reward':>8}  {'Actions (0=GrpA green, 1=GrpB green)':}")
    print("-" * 60)
    try:
        while not done:
            action_dict = policy.predict(
                {tls_id: np.asarray(obs[tls_id], dtype=np.float32) for tls_id in cfg.tls_ids},
                deterministic=True,
            )

            # Count phase switches
            for tls_id, action in action_dict.items():
                if prev_actions[tls_id] != -1 and action != prev_actions[tls_id]:
                    phase_switch_counts[tls_id] += 1
                prev_actions[tls_id] = action

            obs, reward, terminated, truncated, info = env.step(action_dict)
            done = any(terminated.values()) or any(truncated.values())
            total_reward += float(np.mean(list(reward.values()))) if reward else 0.0
            step_count += 1

            if step_count % 20 == 0:
                acts = "  ".join(
                    f"{tid}={'GrpA' if a == 0 else 'GrpB'}" for tid, a in action_dict.items()
                )
                print(f"{step_count:<6} {total_reward:>8.2f}  [{acts}]  switches={dict(phase_switch_counts)}")

            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
    finally:
        print("\n" + "="*40)
        print("🏁 SIMULATION DONE")
        print(f"📊 Total Steps: {step_count}")
        print(f"💰 Total Reward: {total_reward:.2f}")
        print(f"🔄 Phase Switches: {dict(phase_switch_counts)}")
        print("="*40)
        env.close()

if __name__ == "__main__":
    main()