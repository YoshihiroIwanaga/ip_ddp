import matplotlib.pyplot as plt
import numpy as np


def plot_xs(
    xs,
    x_ref,
    x_min,
    x_max,
    horizon,
) -> None:
    n = xs.shape[0]

    fig = plt.figure(figsize=(15, 3.5 * n))
    for i in range(n):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(xs[i, :], "deepskyblue", linestyle="dashed")
        if np.isinf(x_min[i]):
            pass
        else:
            ax.plot((horizon + 1) * [x_min[i]], "black")
        if np.isinf(x_max[i]):
            pass
        else:
            ax.plot((horizon + 1) * [x_max[i]], "black")
        ax.plot((horizon + 1) * [x_ref[i]], "blue")
    plt.show()


def plot_us(
    us,
    u_min,
    u_max,
    horizon,
) -> None:
    m = us.shape[0]

    fig = plt.figure(figsize=(15, 3.5 * m))
    for i in range(m):
        ax = fig.add_subplot(m, 1, i + 1)
        ax.plot(us[i, :], "deepskyblue", linestyle="dashed")
        if np.isinf(u_min[i]):
            pass
        else:
            ax.plot((horizon + 1) * [u_min[i]], "black")
        if np.isinf(u_max[i]):
            pass
        else:
            ax.plot((horizon + 1) * [u_max[i]], "black")
    plt.show()
