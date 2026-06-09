from src.traffic_env.config import build_default_config
from src.traffic_env.envs.multi_agent import MappoTrafficEnv


def main() -> None:
    cfg = build_default_config(
        sumo_cfg_path="sumo_configs/training/sumo_config.sumocfg",
        gui=False,
    )
    env = MappoTrafficEnv(config=cfg, gui=False, use_libsumo=True)
    obs, infos = env.reset()
    print("Initial observation keys:", list(obs.keys()))
    actions = {tls_id: 0 for tls_id in cfg.tls_ids}
    obs, rewards, terms, truncs, infos = env.step(actions)
    print("Reward after one step:", rewards)
    env.close()


if __name__ == "__main__":
    main()