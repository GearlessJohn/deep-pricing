import numpy as np
import os
import matplotlib.pyplot as plt
from implied_vol_codearmo import (
    black_scholes_call,
    vega,
    implied_volatility_call_bisection,
    implied_volatility_call_newton,
    hallerbach_approximation,
)

# Test to verify the functions
# First test to verify the functions


def test1(T=1, S=100, r=0.05, sigma=0.3, start=40, end=160):

    X = np.arange(start, end, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(X/S, [black_scholes_call(S, k, T, r, sigma) for k in X], label="call price")
    ax.plot(X/S, [vega(S, k, T, r, sigma) for k in X], label="vega")

    ax.set_xlabel("K/S")
    ax.set_ylabel("C")
    ax.set_title(f"Call price and vega as function of Spot")

    plt.legend(loc="best")

    plt.savefig("./fig/Call price and vega as function of Spot.png")


# Test for calculs of sigma
def test2(
    K=100,
    T=1 / 12,
    r=0.05,
    sigma=0.25,
    start=10,
    end=200,
    max_iterations=100,
    tol=0.0001,
):

    X = range(start, end, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        X,
        [
            implied_volatility_call_newton(
                black_scholes_call(s, K, T, r, sigma),
                s,
                K,
                T,
                r,
                max_iterations=max_iterations,
                tol=tol,
            )
            for s in X
        ],
        label="newton",
    )

    ax.plot(
        X,
        [
            implied_volatility_call_bisection(
                black_scholes_call(s, K, T, r, sigma),
                s,
                K,
                T,
                r,
                max_iterations=max_iterations,
                tol=tol,
            )
            for s in X
        ],
        label="bisection",
    )

    ax.plot(
        X,
        [
            hallerbach_approximation(black_scholes_call(s, K, T, r, sigma), s, K, T, r)
            for s in X
        ],
        label="hallerbach",
    )

    ax.axhline(sigma, color="r", label="theoretical sigma")

    ax.set_xlabel("S")
    ax.set_ylabel("sigma")
    ax.set_title(f"Implied Volatility Call- max iter {max_iterations}, tol {tol}")

    plt.legend(loc="best")

    plt.savefig(
        f"./fig/Implied Volatility Call-max iter {max_iterations}, tol {tol}.png"
    )



def test3(T=1, S=100, K=100, r=0.05, start=0, end=100):

    X = np.arange(start, end, 1) * 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(X, [black_scholes_call(S, K, T, r, sigma) for sigma in X], label="call price")
    # ax.plot(X/S, [vega(S, K, T, r, sigma) for sigma in X], label="vega")

    ax.set_xlabel("sigma")
    ax.set_ylabel("C")
    ax.set_title(f"Call price as function of sigma")

    plt.legend(loc="best")

    plt.savefig("./fig/Call price as function of sigma.png")



if __name__ == "__main__":
    if not os.path.exists("./fig/"):
        os.mkdir("./fig/")

    test1()
    test2(start=50, end=150)
    # test3()