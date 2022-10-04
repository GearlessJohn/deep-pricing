import matplotlib.pyplot as plt
from implied_vol_codearmo import black_scholes_call, vega

# Test to verify the functions


def test1(T=1, K=100, r=0.05, sigma=0.3, start=10, end=160):

    X = range(start, end, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(X, [black_scholes_call(s, K, T, r, sigma) for s in X], label="call price")
    ax.plot(X, [vega(s, K, T, r, sigma) for s in X], label="vega")

    ax.set_xlabel("S")
    ax.set_ylabel("C")
    ax.set_title(f"Call price and vega as function of Spot")

    plt.legend(loc="best")

    plt.savefig("./fig/Call price and vega as function of Spot.png")


if __name__ == "__main__":
    test1()
