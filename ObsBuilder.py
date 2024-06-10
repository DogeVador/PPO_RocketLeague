from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
import numpy as np


class CustomObsBuilder(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) :
        obs=np.zeros(90)
        for player in state.players:
            obs[0:3] = player.car_data.position
            obs[3:6] = player.car_data.linear_velocity
            obs[6] = np.linalg.norm(player.car_data.linear_velocity)
            obs[7:10] = player.car_data.angular_velocity
            obs[10:13] = player.car_data.euler_angles()
            obs[13] = player.car_data.position[0] - state.ball.position[0]
            obs[14] = player.car_data.position[1] - state.ball.position[1]
            obs[15] = player.car_data.position[2] - state.ball.position[2]
            obs[16] = np.linalg.norm(player.car_data.position - state.ball.position)
            obs[17] = player.car_data.linear_velocity[0] - state.ball.linear_velocity[0]
            obs[18] = player.car_data.linear_velocity[1] - state.ball.linear_velocity[1]
            obs[19] = player.car_data.linear_velocity[2] - state.ball.linear_velocity[2]
            obs[20] = np.linalg.norm(player.car_data.linear_velocity - state.ball.linear_velocity)
            obs[21] = player.has_flip
            obs[22] = player.on_ground
            obs[23] = player.boost_amount
            obs[24] = player.is_demoed

        obs[25:59] = state.boost_pads
        obs[59:62] = state.ball.position
        obs[62:65] = state.ball.linear_velocity
        obs[65] = np.linalg.norm(state.ball.linear_velocity)
        obs[66:69] = state.ball.angular_velocity

        obs[69] = previous_action[0] == 0.0
        obs[70] = previous_action[0] == 1.0
        obs[71] = previous_action[0] == 2.0
        obs[72] = previous_action[1] == 0.0
        obs[73] = previous_action[1] == 1.0
        obs[74] = previous_action[1] == 2.0
        obs[75] = previous_action[2] == 0.0
        obs[76] = previous_action[2] == 1.0
        obs[77] = previous_action[2] == 2.0
        obs[78] = previous_action[3] == 0.0
        obs[79] = previous_action[3] == 1.0
        obs[80] = previous_action[3] == 2.0
        obs[81] = previous_action[4] == 0.0
        obs[82] = previous_action[4] == 1.0
        obs[83] = previous_action[4] == 2.0
        obs[84] = previous_action[5] == 0.0
        obs[85] = previous_action[5] == 1.0
        obs[86] = previous_action[6] == 0.0
        obs[87] = previous_action[6] == 1.0
        obs[88] = previous_action[7] == 0.0
        obs[89] = previous_action[7] == 1.0

        return obs