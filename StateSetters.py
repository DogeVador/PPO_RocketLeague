import numpy as np
from rlgym.utils.state_setters import StateSetter,StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z

class CustomStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):

        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        desired_car_pos = [100, 100, 17]  # x, y, z
        desired_yaw = np.pi / 2

        # Loop over every car in the game.
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = desired_yaw

            elif car.team_num == ORANGE_TEAM:
                # We will invert values for the orange team so our state setter treats both teams in the same way.
                pos = [-1 * coord for coord in desired_car_pos]
                yaw = -1 * desired_yaw

            # Now we just use the provided setters in the CarWrapper we are manipulating to set its state. Note that here we are unpacking the pos array to set the position of
            # the car. This is merely for convenience, and we will set the x,y,z coordinates directly when we set the state of the ball in a moment.
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.01

        # Now we will spawn the ball in the center of the field, floating in the air.
        state_wrapper.ball.set_pos(x=0, y=0, z=CEILING_Z / 2)
        state_wrapper.ball.set_lin_vel(x=1000,y=1000,z=0)