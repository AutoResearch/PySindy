import numpy as np
from scipy.integrate import odeint

from base import Model1D


class Oscillator(Model1D):

    # Input constants
    m: float  # mass (kg)
    L: float  # length (m)
    b: float  # damping value (kg/m^2-s)
    #  g: float  # gravity (m/s^2)
    delta_t: float  # time step size (seconds)
    t_max: float  # max sim time (seconds)
    theta1_0: float  # initial angle (radians)
    theta2_0: float  # initial angular velocity (rad/s)

    def __init__(self, m=1., L=1., b=0.5, delta_t=0.02, t_max=10., theta1_0=np.pi/10, theta2_0=0., noise=0.):
        Model1D.__init__(self)
        self.g = 9.81
        theta_init = (theta1_0, theta2_0)  # initial conditions
        # Get timesteps
        t = np.linspace(0, t_max, int(t_max / delta_t))
        self.t = t

        def init_oscillator(theta_init, t):
            theta_dot_1 = theta_init[1]
            theta_dot_2 = -b / m * theta_init[1] - self.g / L * np.sin(theta_init[0])
            return theta_dot_1, theta_dot_2

        self.X = odeint(init_oscillator, theta_init, t).T\
            + np.random.normal(loc=np.mean(self.y), scale=noise, size=self.y.shape)
