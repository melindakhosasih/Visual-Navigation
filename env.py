import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models, transforms

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
)

from habitat import RLEnv
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut
from habitat.core.dataset import Dataset
from habitat.core.simulator import AgentState, Observations, Simulator

if TYPE_CHECKING:
    from omegaconf import DictConfig

class HabitatEnv(RLEnv):
    def __init__(self, config: "DictConfig", model, dataset: Optional[Dataset] = None, algo: str = None) -> None:
        super(HabitatEnv, self).__init__(config, dataset)
        self.model = model
        self.algo = algo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(pretrained=True).to(self.device)
        self.success_distance = self._env._config.task.measurements.success.success_distance
        self.action_info = None
        self.prev_orien = 0.0
        self.total_reward = 0.0
        self.reward_orien = 0.0

    def get_reward_range(self):
        return -2, 20
    
    def get_reward(self, observations: Observations, *args) -> Any:
        curr_dist, curr_deg = observations["pointgoal_with_gps_compass"]
        # Distance Reward
        self.reward_dist = self._env.get_metrics()["distance_to_goal_reward"]
        self.reward_dist *= 1

        curr_deg = np.rad2deg(abs(curr_deg))
        while curr_deg > 180:
            curr_deg -= 360
        while curr_deg < -180:
            curr_deg += 360
        curr_deg = np.deg2rad(abs(curr_deg))

        # Orientation Reward
        self.reward_orien = self.prev_orien - abs(curr_deg)
        self.reward_orien *= 10
        self.prev_orien = abs(curr_deg)

        # Total Reward
        self.total_reward = self.reward_dist + self.reward_orien

        # Check collision
        collision = self._env.get_metrics()["collisions"]["is_collision"]

        if collision:
            self.total_reward -= 0.2

        if curr_dist < self.success_distance:
            self.total_reward = 20
        elif curr_dist > 5:
            self.total_reward += -0.1
        return self.total_reward

    def get_done(self, observations: Observations) -> bool:
        curr_target_dist = self._env.get_metrics()['distance_to_goal']
        if curr_target_dist < self.success_distance:
            return True
        return False

    def get_info(self, observations) -> Dict[Any, Any]:
        return self.action_info

    def construct_state(self, rp, img):
        rp = np.array([rp[0]/10, np.cos(rp[1]), np.sin(rp[1])])
        norm_img = img.astype(np.float32) / 255.0
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        # relative pose
        return rp

        # resnet
        # img_ts = preprocess(img)
        # img_ts = img_ts.unsqueeze(0).to(self.device)
        # resnet_img = self.resnet(img_ts).cpu()
        # np_img = resnet_img.detach().numpy()
        # np_img = np_img.reshape(-1)
        # state = np.concatenate((rp, np_img))
        # return state

        # img
        # norm_img = norm_img.flatten()   # shape (196608,)
        # state = np.concatenate((rp, norm_img))  # shape (196611,)
        # return state
            
    def reset(self) -> Observations:
        """
            ### return rp (distance, radian), obs
        """
        obs = super().reset()
        img = obs["rgb"]
        rp = obs["pointgoal_with_gps_compass"]

        self.total_reward = 0.0
        self.reward_orien = 0.0
        self.prev_orien = abs(rp[1])
        return self.construct_state(rp, img), obs
    
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        observations, reward, done, info = super().step(*args, **kwargs)
        rp = observations["pointgoal_with_gps_compass"]
        img = observations["rgb"]
        if self._env._episode_over or done:
            done = True

        return self.construct_state(rp, img), reward, done, info, observations
    
    def translate_action(self, action):
        # Debug
        # print("action", action.shape)

        # move forward range (0, 0.1) meter
        velocity = (action[0] + 1) / 2
        forward = round(velocity / 0.01)
        
        # turn range (-10, 10) degree
        turn = abs(round(action[1] / 0.1))

        action_lists = []
        # turn left
        if action[1] < 0:
            action_lists.extend(["turn_left"] * turn)
        # turn right
        elif action[1] > 0:
            action_lists.extend(["turn_right"] * turn)
        # move forward
        action_lists.extend(["move_forward"] * forward)

        self.action_info = f"action[0]: {forward * 0.001:.3f}m, action[1]: {round(action[1] / 0.1)}"
        return tuple(action_lists)

    def plot_fig(self, overall_succ_rate, succ_rate_split, path, eval_eps):
        plt.plot(overall_succ_rate, label="Overall Training Succ")
        plt.plot(succ_rate_split, label=f"Avg of {eval_eps} Episodes", linestyle='--')

        plt.xlabel('Episode')
        plt.ylabel('Succ')
        plt.legend()

        plt.savefig(f'{path}/training.png')
        plt.close()   
    
    def train(self, out_path, n_epi=1000, batch_size=64, n_eval=50):
        torch.cuda.empty_cache()
        total_step = 0
        max_success_rate = 0
        success_count = 0
        total_succ_rate = []
        overall_succ_rate = []
        succ_rate_split = []
        
        for eps in range(n_epi):
            state, _ = self.reset()
            # break
            # Debug
            # print("="*50)
            # print(state)
            # print("="*50)

            step = 0
            loss_a = loss_c = 0
            total_reward = 0.

            while True:
                # Choose action
                action = self.model.choose_action(state, eval=False)

                # Debug
                # print(action)
                # print("state.shape", state["rgb"].shape)

                # Step
                state_next, reward, done, info, _ = self.step({
                    "action": self.translate_action(action)
                })
            
                # Debug
                # print("done", done, "over", self._env._episode_over)
                # print(state_next, info)

                # Store
                end = 0 if done else 1
                self.model.store_transition(state, action, reward, state_next, end)

                # Learn
                loss_a = loss_c = 0.
                if total_step > batch_size:
                    loss_a, loss_c = self.model.learn()

                step += 1
                total_step += 1
                total_reward += reward

                if self.algo == "ddpg":
                    print(f"\rEps:{eps:3d} /{step:4d} /{total_step:6d}| "
                        f"V:{action[0]:+.2f}| W:{action[1]:+.2f}| "
                        f"R:{reward:+.2f}| "
                        f"Loss:[A>{loss_a:+.2f} C>{loss_c:+.2f}]| "
                        f"Epsilon: {self.model.epsilon:.3f}| "
                        f"Ravg:{total_reward/step:.2f}", end='')
                elif self.algo == "sac":
                    print(f"\rEps:{eps:3d} /{step:4d} /{total_step:6d}| "
                        f"V:{action[0]:+.2f}| W:{action[1]:+.2f}| "
                        f"R:{reward:+.2f}| "
                        f"Loss:[A>{loss_a:+.2f} C>{loss_c:+.2f}]| "
                        f"Alpha: {self.model.alpha:.3f}| "
                        f"Ravg:{total_reward/step:.2f}", end='')
                else:
                    assert self.algo is None, "Algorithm doesn't exist"

                state = state_next.copy()
                if done:
                    # Count the successful times
                    if reward > 5:
                        success_count += 1
                        total_succ_rate.append(1)
                    else:
                        total_succ_rate.append(0)
                    print()
                    break
            
            if not self._env.episode_over:
                self.step({"action": "stop"}) 

            overall_succ_rate.append(np.mean(total_succ_rate))
            succ_rate_split.append(np.mean(total_succ_rate[-n_eval:]))
            
            self.plot_fig(overall_succ_rate, succ_rate_split, out_path, n_eval)

            if eps>0 and eps%n_eval==0:
                # Sucess rate
                success_rate = success_count / (n_eval+1)
                success_count = 0

                # Save the best model
                if success_rate >= max_success_rate:
                    max_success_rate = success_rate
                print("Save model to " + out_path+"models/")
                self.model.save_load_model("save", out_path+"models/", eps)
                print(f"Success Rate (current/max): {success_rate}/{max_success_rate}")
                # output video
                self.eval(self.model, total_eps=5, video_path=out_path+"videos/", video_name=f"{self.algo}_"+str(eps).zfill(4), message=True)
    
    def eval(self, model, video_path, video_name, total_eps=3, message=False):
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        for eps in range(total_eps):
            state, obs = self.reset()
            step = 0
            total_reward = 0.

            # Render
            frame = self.render(obs)
            # Add frame to vis_frames
            vis_frames = [frame]

            while True:
                # Choose action
                action = self.model.choose_action(state, eval=False)

                # Step
                state_next, reward, done, info, obs = self.step({"action": self.translate_action(action)})
                total_reward += reward

                # Render
                frame = self.render(obs)
                # Add frame to vis_frames
                vis_frames.append(frame)

                if message:
                    print(f"\rEps:{eps:2d} /{step:4d} | action:{action[0]:+.2f}| "
                          f"R:{reward:+.2f} | Total R:{total_reward:.2f}", end='')

                state = state_next.copy()
                step += 1

                if done:
                    # Count the successful times
                    if message:
                        print()
                    break
            
            # Create video from images and save to disk
            # Render Last Frame with SPL and Succ
            if not self._env.episode_over:
                _, _, _, _, obs = self.step({"action": "stop"}) 
                frame = self.render(obs)
                vis_frames.append(frame)

            print("Save video...")
            images_to_video(
                vis_frames, video_path, f"{video_name}_{str(eps).zfill(2)}", fps=6, quality=9
            )
            vis_frames.clear()
            # Display video
            # vut.display_video(f"{video_path}/{video_name}_{str(eps).zfill(2)}.mp4")
    
    def render(self, obs):
        map_info = self._env.get_metrics()
        # Concatenate RGB observation and topdown map into one image
        if "depth" in obs:
            obs.pop("depth")
        frame = observations_to_image(obs, map_info)
        # Remove top_down_map from metrics
        map_info.pop("top_down_map")

        # Update distance reward
        map_info["distance_to_goal_reward"] = self.reward_dist

        # Add orientation reward
        map_info["orien_to_goal_reward"] = self.reward_orien

        # Add total reward
        map_info["total_reward"] = self.total_reward

        # Overlay numeric metrics onto frame
        frame = overlay_frame(frame, map_info)
        return frame

if __name__ == "__main__":
    import os
    import cv2
    import habitat
    from habitat.config.default_structured_configs import (
        FogOfWarConfig,
        TopDownMapMeasurementConfig
    )

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    config = habitat.get_config(
        config_path="./demo.yaml"
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

    env = HabitatEnv(model=None, config=config, algo=None)
    total_step = 0
    for eps in range(1200):
        env.reset()
        print(f"EP: {eps}")
        while not env._env._episode_over:
            # Choose action
            key = cv2.waitKey(0)
            if key == ord("w") or key == ord("W"):
                # print("move forward")
                action = [1, 0]
            elif key == ord("a") or key == ord("A"):
                # print("turn left")
                action = [-1, -1]
            elif key == ord("s") or key == ord("S"):
                # print("move backward")
                action = [-1, 0]
            elif key == ord("d") or key == ord("D"):
                # print("turn right")
                action = [-1, 1]
            else:
                action = [-1, 0]

            # Step
            state_next, reward, done, info, obs = env.step({
                "action": env.translate_action(action)
            })

            if key == 27: # ESC button
                env.step({"action": "stop"}) 

            frame_bgr = env.render(obs)
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cv2.imshow("RGB", frame)
    cv2.destroyAllWindows()