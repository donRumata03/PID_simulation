import numpy as np


class model:
    integral = 0
    e_last : float

    kp : float
    kd : float
    ki : float

    target_t : float

    index = 0

    def __init__(self, kp, kd, ki, target_t, dt):
        self.kp = kp
        self.kd = kd / dt
        self.ki = ki * dt
        self.target_t = target_t

    def step(self, t):
        e = t - self.target_t
        self.integral += e

        diff = e - self.e_last if self.index != 0 else 0
        self.index += 1
        self.e_last = e

        p = - self.kp * e
        d = - self.kd * diff
        i =  - self.ki * self.integral

        print("T:", t, "E:", e, "P:", p, "; I:", i, "; D:", d)

        res = p + i + d

        print("Res:", res)

        return res

