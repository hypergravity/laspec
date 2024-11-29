import numpy as np
import matplotlib.pyplot as plt
from laspec.extern.interpolate import SmoothSpline

x = np.linspace(0, 1000, 100)
y = -((x - 500) ** 2) / 1e5 + np.random.normal(loc=0, scale=10, size=x.shape)


def draw_smooth_spline(x, y, p: float = 0.1):
    return SmoothSpline(x, y, p=p)(x)


plt.plot(x, y, label="original data")
for p in np.logspace(-6, -2, 3):
    plt.plot(x, SmoothSpline(x, y, p=p)(x), label=f"p={p}")
plt.legend()
plt.show()
