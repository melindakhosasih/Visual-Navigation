import os
import argparse

import habitat
from habitat.config.default_structured_configs import (
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)

import model.model as model
from algo.ddpg import DDPG
from algo.sac import SAC
from env import HabitatEnv

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, required=True, help="Experiment's Title")
    parser.add_argument("--algo", default="ddpg", type=str, help="RL Algorithm")
    parser.add_argument("--config", default="./test.yaml", type=str, help="habitat config's path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()

    out_path = f"./out/{args.algo}/{args.title}/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if args.algo == "ddpg":
        model = DDPG(
            model = [model.PolicyNet, model.QNet],
                learning_rate = [0.0001, 0.0001],
                reward_decay = 0.99,
                memory_size = 10000,
                batch_size = 64
        )
    elif args.algo == "sac":
         model = SAC(
            model = [model.PolicyNetGaussian, model.QNet],
            n_actions = 2,
            learning_rate = [0.0001, 0.0001],
            reward_decay = 0.99,
            memory_size = 10000,
            batch_size = 64,
            alpha = 0.1,
            auto_entropy_tuning=True
        )
    else:
        assert args.algo is None, "Algorithm doesn't exist"

    config = habitat.get_config(
        config_path=args.config
    )

    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                )
            }
        )

    env = HabitatEnv(model=model, config=config, algo=args.algo)
    env.train(out_path=out_path, batch_size=64, n_epi=2501, n_eval=50)
    print("Finished!!!")
