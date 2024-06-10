from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
import rlgym.utils.common_values as cm
from rlgym.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward, EventReward,AlignBallGoal
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward,FaceBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, TouchBallReward, VelocityBallToGoalReward,VelocityReward
from  rlgym.utils.reward_functions.combined_reward import CombinedReward
import numpy as np



class CustomReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward=CombinedReward.from_zipped(
            (TouchBallReward(),0.5),
            (SaveBoostReward(),0.05),
            (LiuDistanceBallToGoalReward(),0.05),
            (LiuDistancePlayerToBallReward(),0.2),
            (FaceBallReward(),0.1),
            (AlignBallGoal(),0.05),
            (VelocityPlayerToBallReward(),0.2),
            (VelocityReward(),0.1),
            (VelocityBallToGoalReward(),0.05)
        ).get_reward(player,state,previous_action)

        return reward

class SimpleReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward=CombinedReward.from_zipped(
            (LiuDistancePlayerToBallReward(),0.5),
            (VelocityReward(),0.5),
        ).get_reward(player,state,previous_action)

        return reward
