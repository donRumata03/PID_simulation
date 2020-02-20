def sgn(x):
    return 1 if x > 0 else (0 if x == 0 else -1)

def abs_cut(val, _max):
    return sgn(val) * min(_max, abs(val))

def dual_abs_cut(val, _min, _max):
    if val < _min:
        return _min
    elif val < _max:
        return val
    return _max

class heater:
    max_power : float
    temperature : float
    diff_coeff : float
    max_temp : float


    def __init__(self, max_power, diff_coeff : float, max_temp : float):
        self.max_power = max_power
        self.temperature = 0
        self.diff_coeff = diff_coeff
        self.max_temp = max_temp

    def execute(self, val):
        # print("Heating:", self.diff_coeff * sgn(val) * min(self.max_power, abs(val)), "Temperature:", self.temperature)
        self.temperature += self.diff_coeff * abs_cut(val, self.max_temp)
        # self.temperature =  abs_cut(self.temperature, self.max_temp)
        self.temperature = dual_abs_cut(self.temperature, -self.max_temp / 3, self.max_temp)
        return self.temperature