from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Optional

import habitat
from habitat.config.default_structured_configs import MeasurementConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Define the measure and register it with habitat
# By default, the things are registered with the class name
@habitat.registry.register_measure
class OrienToGoalReward(habitat.Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `rad(abs(angle))` towards goal.
    """

    cls_uuid: str = "orien_to_goal_reward"

    def __init__(self, sim, config: "DictConfig", **kwargs: Any):
        # This measure only needs the config
        self._config = config
        self._previous_orien: Optional[float] = None
        super().__init__()

    # Defines the name of the measure in the measurements dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # This is called whenever the environment is reset
    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        # Our measure always contains all the attributes of the episode
        self._metric = vars(episode).copy()
        # But only on reset, it has an additional field of my_value
        self._metric["my_value"] = self._config.VALUE

    # This is called whenever an action is taken in the environment
    def update_metric(self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any):
        # Now the measure will just have all the attributes of the episode
        self._metric = vars(episode).copy()

@registry.register_measure
class DistanceToGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "distance_to_goal_reward"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = -(distance_to_target - self._previous_distance)
        self._previous_distance = distance_to_target


# define a configuration for this new measure
@dataclass
class OrienToGoalRewardMeasurementConfig(MeasurementConfig):
    r"""
    In Navigation tasks only, measures a reward based on the orientation towards the goal.
    The reward is `rad(abs(angle))` towards goal.
    """
    type: str = "OrienToGoal"