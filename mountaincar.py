import numpy as np


class MountainCar():
    """A mountain-car problem.

    Usage:
        >>> mc = MountainCar()

        Set the agent to apply a rightward force (positive in x)
        >>> mc.apply_force(+1) # the actual value doesn't mattter, only the sign

        Run an "agent time step" of 1s with 0.01 s integration time step
        >>> mc.simulate_timesteps(n = 100, dt = 0.01)

        Check the state variables of the agent, and the reward
        >>> print mc.x, mc.x_d, mc.R

        At some point, one might want to reset the position/speed of the car
        >>> mc.reset()
    """

    def __init__(self, g=10.0, d=100.0, H=10., m=10.0, F=10.0, R=1.,
                 T=0.0):
        self.g = g
        self.d = d  # minima location
        self.H = H  # height of the saddle point
        self.m = m  # mass of the car
        self.F = F  # amplitude of the maximum force applied by the engine
        self.R = R  # value of the reward
        self.T = T  # x-axis threshold for the obtention of reward

        # reset the car variables
        self.reset()

    def reset(self):
        """Reset the mountain car to a random initial position.
        """

        # set position to range [-130; -50]
        self.x = np.random.uniform(-130, -50)

        # set v to range [-5; 5]
        self.vx = np.random.uniform(-5, 5)
        # reset reward
        self.r = 0.
        # reset time
        self.t = 0.
        # reset applied force
        self.f = 0.

    def apply(self, action):
        """
        Act on mountain car:
            -1 -> left
             0 -> no push
            +1 -> right
        """
        self.f = np.clip(action, -self.F, self.F)

    def _h(self, x):
        """
        height(x)

        """
        return (x - self.d)**2 * (x + self.d)**2 / ((self.d**4/self.H)+x**2)

    def _dhdx(self, x):
        """
        dheight(x) / dx
        """
        c = self.d**4/self.H
        n = 2 * x * (x**2 - self.d**2) * (2*c + self.d**2 + x**2)
        d = (c+x**2)**2
        return n / d

    def _d2hdx2(self, x):
        """
        d^2height(x) / dx^2
        """
        c = self.d**4/self.H
        n = -2 * c**2 * (self.d**2 - 3*x**2)
        n += c * (-self.d**4 + 6*self.d**2 * x**2 + 3*x**4)
        n += 3 * self.d**4 * x**2
        n += x**6
        n *= 2
        d = (c + x**2)**3
        return n / d

    def _E(self, x, vx):
        """
        Energy at position x, with speed v
        """
        # note that v and x dot are not the same: v includes the y direction!
        return self.m*(self.g*self._h(x)+0.5*(1+self._dhdx(x)**2)*vx**2)

    def act(self, action):
        self.apply(action)
        self.update()
        return self.r

    def update(self, n=100, dt=0.01):
        """
        Update state of the moutain car
        """

        for i in range(n):
            self._update(dt)

        self.t += n*dt

        # check for rewards
        self.r = self._get_reward()

    def _update(self, dt):
        """
        Short hand update
        """

        # compute d2x/dt2 (horiz. acceleration)
        alpha = np.arctan(self._dhdx(self.x))
        ax = self.f / self.m
        ax -= np.sin(alpha) * (self.g + self._d2hdx2(self.x) * self.vx**2)
        ax *= np.cos(alpha)

        # update
        self.x  += self.vx * dt + 0.5 * ax * dt**2
        self.vx += ax * dt

    def _get_reward(self):
        """Check for and return reward.
        """

        # if there's already a reward, we stick to it
        if self.r > 0.0:
            return self.r
        # have we crossed the threshold?
        if self.x >= self.T:
            return self.R

        # else no reward
        return -0.1

