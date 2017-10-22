"""Portfolio analyzer
The analyzer compute daily portfolio value given stock prices,
allocations and starting value.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from util import get_data, plot_data


def get_portfolio_value(prices, allocs, start_val=1):
    """
    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)
    Returns
    -------
        port_val: daily portfolio value
    """
    port = prices.copy()
    for col in port:
        port[col]/=port[col].iloc[0]
    daily_port_val = port.dot(allocs)
    port_val = daily_port_val * start_val
    return port_val

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.
    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)
    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    df=port_val.copy()
    daily_ret = (df/ df.shift(1))-1
    daily_ret.fillna(0)
    cum_ret=(df[-1]-df[0])/df[0]

    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    sharpe_ratio=((252)**0.5)*avg_daily_ret/std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    df2=df.copy()
    for col in df2:
        df2[col]/=df2[col].iloc[0]
    ax=df2.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot()
    fig=plt.gcf()
    fig.savefig('comparion.png')


def assess_portfolio(start_date, end_date, symbols, allocs, start_val=1):
    """Simulate and assess the performance of a stock portfolio."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs, start_val)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret

    # Compare daily portfolio value with SPY using a normalized plot
    df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and SPY")


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2010-12-31'

    symbol_allocations = OrderedDict([('GOOG', 0.2), ('AAPL', 0.2), ('GLD', 0.4), ('XOM', 0.2)])  # symbols and corresponding allocations
    symbols = symbol_allocations.keys()  # list of symbols, e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = symbol_allocations.values()  # list of allocations, e.g.: [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  # starting value of portfolio
    # Assess the portfolio
    assess_portfolio(start_date, end_date, symbols, allocs, start_val)
