import random
from matplotlib import pyplot as plt
import numpy as np

import graphic_smoother


class Vector(list):
    def __init__(self, *el):
        for e in el:
            self.append(e)

    def __add__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] + other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self + other

    def __sub__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] - other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self - other

    def __mul__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] * other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self * other

    def __truediv__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] / other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self / other

    def __pow__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] ** other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self ** other

    def __mod__(self, other):
        return sum((self - other) ** 2) ** 0.5

    def mod(self):
        return self % Vector.emptyvec(len(self))

    def dim(self):
        return len(self)

    def __str__(self):
        if len(self) == 0:
            return "Empty"
        r = [str(i) for i in self]
        return "< " + " ".join(r) + " >"

    def _ipython_display_(self):
        print(str(self))

    @staticmethod
    def emptyvec(lens=2, n=0):
        return Vector(*[n for i in range(lens)])

    @staticmethod
    def randvec(dim):
        return Vector(*[random.random() for i in range(dim)])


class Point:
    def __init__(self, coords, mass=1.0, q=1.0, speed = None, ** properties):
        self.coords = coords
        if speed is None:
            self.speed = Vector(*[0 for i in range(len(coords))])
        else:
            self.speed = speed
        self.acc = Vector(*[0 for i in range(len(coords))])
        self.mass = mass
        self.__params__ = ["coords", "speed", "acc", "q"] + list(properties.keys())
        self.q = q
        for prop in properties:
            setattr(self, prop, properties[prop])


    def move(self, dt):
        self.coords = self.coords + self.speed * dt


    def accelerate(self, dt):
        self.speed = self.speed + self.acc * dt


    def accinc(self, force):
        self.acc = self.acc + force / self.mass


    def clean_acc(self):
        self.acc = self.acc * 0


    def __str__(self):
        r = ["Point {"]
        for p in self.__params__:
            r.append("  " + p + " = " + str(getattr(self, p)))
        r += ["}"]
        return "\n".join(r)


    def _ipython_display_(self):
        print(str(self))



class InteractionField:
    def __init__(self, F):
        self.points = []
        self.F = F

    def move_all(self, dt):
        for p in self.points:
            p.move(dt)

    def intensity(self, coord):
        proj = Vector(*[0 for i in range(coord.dim())])
        single_point = Point(Vector(), mass=1.0, q=1.0)
        for p in self.points:
            if coord % p.coords < 10 ** (-10):
                continue
            d = p.coords % coord
            fmod = self.F(single_point, p, d) * (-1)
            proj = proj + (coord - p.coords) / d * fmod
        return proj

    def step(self, dt):
        self.clean_acc()
        for p in self.points:
            p.accinc(self.intensity(p.coords) * p.q)
            p.accelerate(dt)
            p.move(dt)

    def clean_acc(self):
        for p in self.points:
            p.clean_acc()

    def append(self, *args, **kwargs):
        self.points.append(Point(*args, **kwargs))

    def gather_coords(self):
        return [p.coords for p in self.points]

def make_noise(beg, end, step, target_max, distr = 100, percent_sigma = 0.1, target_num = 1000):
    xs = np.arange(beg, end, step)
    raw_noise = np.random.normal(0, distr, xs.shape)

    print("Raw noise")

    smoothed = np.array(graphic_smoother.smooth_graph(list(zip(xs, raw_noise)), percent_sigma=percent_sigma, percent_frame_size=0.1)).T[1]
    this_max = max(abs(smoothed))
    print(this_max)
    smoothed *= target_max / this_max

    # smoothed_xs = np.arange(beg, end, (end - beg) / smoothed.shape[0])[:smoothed.shape[0]]

    """
    plt.plot(xs, raw_noise)
    plt.plot(smoothed_xs, smoothed)
    plt.show()
    """
    return  smoothed

if __name__ == "__main__":
    make_noise(0, 3000, 0.7, 1000, percent_sigma=0.01)

"""
u = InteractionField(lambda p1, p2, r: 300000 * -p1.q * p2.q / (r ** 4 + 0.1))
for i in range(200):
    u.append(Vector.randvec(5) * 10, q=random.random() - 0.5)

pure = np.linspace(-1, 1, 100)

noise = np.random.normal(0, 1, pure.shape)
signal = pure + graphic_smoother.smooth_graph(list(noise), 1, 1)

plt.plot(pure, signal)
plt.show()

"""

"""

velmod = 0
velocities = []
for i in range(100):
    u.step(0.05)
    velmod = sum([p.speed.mod() for p in u.points])   # Добавляем сумму модулей скоростей всех точек
    velocities.append(velmod)
plt.plot(velocities)
plt.show()
"""