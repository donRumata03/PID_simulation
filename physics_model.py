from PID_model import model
from heater_model import heater


class physics:
    dt : float

    t0 : float
    t_okr : float
    t : float
    max_t : float

    lose_coeff : float

    h : heater
    m : model

    def __init__(self, h : heater, m  : model, t0 : float, dt : float, lose_coeff : float, t_okr : float = None, max_t = 99.7):
        self.h = h
        self.m = m

        self.t0 = t0
        self.t = t0
        if t_okr is not None:
            self.t_okr = t_okr
        else:
            self.t_okr = t0

        self.lose_coeff = lose_coeff
        self.dt = dt
        self.max_t = max_t


    def step(self):
        PID_val = self.m.step(self.t)
        heater_power = self.h.execute(PID_val)

        self.t += heater_power * self.dt

        power_lose = self.lose_coeff * (self.t - self.t_okr)
        self.t -= power_lose * self.dt

        total_power_now = heater_power - power_lose

        print(power_lose)

        self.t = min(self.t, self.max_t)

        return self.t, heater_power, power_lose, total_power_now


