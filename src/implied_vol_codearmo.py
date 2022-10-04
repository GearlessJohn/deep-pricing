# Imports
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Gaussian distributions, as reference
N_prime = norm.pdf
N = norm.cdf


def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)
    return call

def vega(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    #see hull derivatives chapter on greeks for reference
    vega = S * N_prime(d1) * np.sqrt(T)
    return vega



def implied_volatility_call_newton(C, S, K, T, r, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3
    
    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C
        
        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma

def implied_volatility_call_bisection(C, S, K, T, r, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''
    
    ### assigning initial volatility estimate for input in Newton_rap procedure
    a = 0
    b = 1
    sigma = 0.5
    
    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C
        
        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            break

        ### use bisection method to update the estimate
        if diff < 0 :
            if black_scholes_call(S, K, T, r, b) < C:
                a = b
                b = 2 * b
            a = sigma
            sigma = 0.5*(a+b)
        if diff > 0 :
            b = sigma
            sigma = 0.5*(a+b)
            
    return sigma
    
    
# Testing those functions with some data

# First test to verify the functions
def test1(T = 1, K = 100,  r = 0.05, sigma = 0.3, start=10, end=160):
    
    X =  range(start, end, 1)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(X, [black_scholes_call(s, K, T, r, sigma) for s in X], label="call price")
    ax.plot(X, [vega(s, K, T, r, sigma) for s in X], label="vega")
    
    ax.set_xlabel('S')
    ax.set_ylabel('C')
    ax.set_title(f"Call price and vega as function of Spot")

    plt.legend(loc='best')

    plt.savefig('./fig/Call price and vega as function of Spot.png')

# Test for implied volatility
def test2(K=100, T=1/12, r=0.05, sigma=0.25, start=10, end=200, max_iterations=100, tol=0.0001):
    
    X = range(start, end, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(X, [implied_volatility_call_newton(black_scholes_call(s, K, T, r, sigma), s, K, T, r, max_iterations=max_iterations, tol=tol) for s in X], label="newton")
    ax.plot(X, [implied_volatility_call_bisection(black_scholes_call(s, K, T, r, sigma), s, K, T, r, max_iterations=max_iterations, tol=tol) for s in X], label="bisection")
    ax.axhline(sigma, color="r", label="theoretical sigma")   
    
    ax.set_xlabel('S')
    ax.set_ylabel('sigma')
    ax.set_title(f"Implied Volatility Call- max iterations {max_iterations}")

    plt.legend(loc='best')

    plt.savefig(f'./fig/Implied Volatility Call-iterations {max_iterations}.png')
    


if __name__ == "__main__":
    # test1()
    test2(max_iterations=1000, tol)

