import os
import numpy as np
import matplotlib.pyplot as plt


def Sigma(t, s, eta, lam):
    return (
        eta**2
        / (2 * lam)
        * np.exp(-lam * (s + t))
        * (np.exp(2 * lam * min(s, t)) - 1)
    )


def covariance(T, n, eta, lam):
    times = np.linspace(0, T, n + 1)[1:]

    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = Sigma(times[i], times[j], eta=eta, lam=lam)

    return cov


def XT(T, n, eta, lam):

    cov = covariance(T, n, eta, lam)
    L = np.linalg.cholesky(cov)

    Zns = np.random.multivariate_normal(np.zeros(n), np.eye(n), 1)

    Xns = np.append([0], L @ Zns.T)

    return Xns


def plot_XT(lam, eta, T=1.5, n=30, repeat=1000):

    times = np.linspace(0, T, n + 1)

    sample = np.array([XT(lam=lam, eta=eta, T=T, n=n) for i in range(repeat)])

    mean = np.mean(sample[:, -1])
    var = np.var(sample[:, -1])

    print(f"X_T: mean={mean:3.2f} var={var:3.2f} ({Sigma(t=T,s=T, eta=eta, lam=lam)})")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(times, sample.T, "-")

    ax.set_xlabel("time t")
    ax.set_ylabel("${X_T}$")
    ax.set_title(
        f"Sample of 1000 copies of the discrete path Xt with lam={lam} and  eta={eta}"
    )

    plt.savefig(f"./fig/Xt with lam {lam}, eta {eta}.png")


if __name__ == "__main__":
    if not os.path.exists("./fig/"):
        os.mkdir("./fig/")

    plot_XT(lam=1, eta=1, repeat=1000)
