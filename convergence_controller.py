import math

from graphic_smoother import *

class clever_convergence_controller:
    array = []
    this_derivatives = []
    previous_derivatives = []
    fractions = []
    precision : float

    def __init__(self, precision : float):
        self.precision = precision

    def push_back(self, val : float):
        self.array.append(val)

    def predict(self) -> bool: # False => no convergence!
        if len(self.array) <= 2:
            self.fractions.append(0)
            self.previous_derivatives.append(1)
            self.this_derivatives.append(1)
            return False
        valuable_samples = self.array[-30:]
        # this_derivative = abs(self.array[-1] - self.array[-2])
        raw_prev_derivatives = [abs(valuable_samples[_i - 1] - valuable_samples[_i]) for _i in range(1, len(valuable_samples))]
        smoothed_derivatives = smooth_graph(list(enumerate(raw_prev_derivatives)), 0.5, 1)
        previous_derivative : float = smoothed_derivatives[0][1]
        this_derivative : float = smoothed_derivatives[-1][1]
        # print(this_derivative, raw_prev_derivatives)
        # print("Previous:", previous_derivative, "Actual:", this_derivative, "Fraction:", previous_derivative / this_derivative)
        self.fractions.append(1 / abs(previous_derivative - this_derivative))
        self.this_derivatives.append(this_derivative)
        self.previous_derivatives.append(previous_derivative)
        return this_derivative < self.precision


def anal_test_func(x : float) -> float:
    return  20 - 100 / x + 100 * (random.random() * 1 / x)


if __name__ == '__main__':
    c = clever_convergence_controller(2)

    xs = np.arange(1, 100, 1)
    func_vals = []

    for i in xs:
        c.push_back(anal_test_func(i))
        # print(c.array[-1], c.predict())
        c.predict()
        func_vals.append(anal_test_func(i))

    plt.plot(xs, func_vals)
    plt.plot(xs, c.fractions)
    plt.plot(xs, c.previous_derivatives)
    plt.plot(xs, c.this_derivatives)
    plt.show()

# TODO!!!