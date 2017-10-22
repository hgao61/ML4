"""portfolio optimizer
The find_optimal_allocations func is aiming at finding the optimal allocations for a given set of stocks
 based on the optimization maximizing for Sharpe ratio.

The input is a list of interested symbols, start and end dates
The return will be float numbers in the format of one-dimensional numpy array
that represents the allocations to each of the equities.
"""
import pandas as pd
import numpy as np

from util import get_data, plot_data
from analysis import get_portfolio_value, get_portfolio_stats

import scipy.optimize as sco


def min_func_sharpe(weights):
    return -statistics(weights)

def statistics(weights):
    weights = np.array(weights)
    port_val=get_portfolio_value(df,weights,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)
    return sharpe_ratio

def find_optimal_allocations(prices):
    """Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio.
    Input
    ----------
        prices: daily prices for each stock in portfolio
    Output
    -------
        allocs: optimal allocations, as fractions that sum to 1.0
    """
    new_prices = prices.copy()
    noa = len(new_prices.keys())
    global df
    df = prices.copy()
    cons =({ 'type': 'ineq', 'fun': lambda inputs: 1.0 - np.sum(abs(inputs)) })
    bnds = tuple((0,1) for x in range(noa))
    weights = np.random.random(noa)
    weights/=np.sum(weights)
    allocs= sco.minimize(min_func_sharpe, noa * [1. /noa,], method='SLSQP', bounds=bnds, constraints=cons)
    return allocs.x


def optimize_portfolio(start_date, end_date, symbols):
    """Simulate and optimize portfolio allocations."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get optimal allocations
    allocs= find_optimal_allocations(prices)
    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0

    # Get daily portfolio value (already normalized since we use default start_val=1.0)
    port_val = get_portfolio_value(prices, allocs)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Optimal allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret

    # Compare daily portfolio value with normalized SPY
    normed_SPY = prices_SPY / prices_SPY.ix[0, :]
    df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_data(df_temp, title="Daily Portfolio Value and SPY")


if __name__ == "__main__":
    # Define input parameters
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    symbols = ['GOOG', 'AAPL', 'GLD', 'HNZ']  # list of symbols
    # Optimize portfolio
    optimize_portfolio(start_date, end_date, symbols)
