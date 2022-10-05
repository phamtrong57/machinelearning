import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import torch



def f1(x):
    return x[0] ** 2 + x[1] ** 2 + x[0] * x[1]


def sgd_optimize(f, init, lr, max_iter=100, **params):
    x = torch.tensor(init, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr, **params)

    xs = [x.clone()]
    x_old = x.clone()
    for i in range(max_iter):
        # 関数値を計算する。
        y = f(x)
        # 勾配を計算する。
        optimizer.zero_grad()
        y.backward()
        # 更新する。
        optimizer.step()

        if (x - x_old).norm() < 0.01:
            break  # 収束した場合は途中で抜ける
        x_old = x.clone()

        xs.append(x.clone())

    xs = torch.stack(xs).detach().numpy()

    return xs

xs = sgd_optimize(f1, init=[3, 7], lr=0.1)

def draw_history(f, xs, elev=70, azim=-70):
    fig = plt.figure(figsize=(14, 6))

    X1, X2 = np.mgrid[-10:11, -10:11]
    Y = f((X1, X2))  # 各点での関数 f の値を計算する。

    ys = [f(x) for x in xs]

    # 勾配のベクトル図を作成する。
    ax1 = fig.add_subplot(121)
    ax1.set_title("Contour")
    ax1.set_xticks(np.arange(-10, 11, 2))
    ax1.set_yticks(np.arange(-10, 11, 2))
    ax1.set_xlabel("$x$", fontsize=15)
    ax1.set_ylabel("$y$", fontsize=15)
    ax1.grid()
    ax1.plot(xs[:, 0], xs[:, 1], "ro-", mec="b", mfc="b", ms=4)
    contours = ax1.contour(X1, X2, Y, levels=15)
    ax1.clabel(contours, inline=1, fontsize=10, fmt="%.2f")

    # グラフを作成する。
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Surface")
    ax2.set_xlabel("$x$", fontsize=15)
    ax2.set_ylabel("$y$", fontsize=15)
    ax2.set_zlabel("$z$", fontsize=15)
    ax2.plot(xs[:, 0], xs[:, 1], ys, "ro-", mec="b", mfc="b", ms=4)
    ax2.plot_surface(X1, X2, Y, alpha=0.3, edgecolor="black")
    ax2.view_init(elev=elev, azim=azim)

    plt.show()


draw_history(f1, xs)