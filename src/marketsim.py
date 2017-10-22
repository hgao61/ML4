"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def listworkingdays(symbol):
    orders_file = os.path.join("data", symbol+".csv")
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Open','High','Low','Close','Volume','Adj Close'], header =0)
    df=df.to_dict('list')
    return df["Date"]

def checkstockprice(date,symbol):

    orders_file = os.path.join("data", symbol+".csv")
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Open','High','Low','Close','Volume','Adj Close'], header =0)
    df=df.to_dict('records')
    #print df
    for dicts in df:
        if dicts["Date"] == date:
            #print dicts["Adj Close"]
            return dicts["Adj Close"]


def tradebookupdate(tradebook, symbol, numberofshares):
    if symbol in tradebook:
        #print "symbol",symbol
        #print "value",tradebook[symbol]
        tradebook[symbol] = tradebook[symbol] + numberofshares
    else:
        tradebook[symbol] = numberofshares
    return

def tradebooktotal(tradebook,date):
   sum = 0
   for (k,v) in tradebook.items():
        if k == 'CASH':
            sum = sum + v
        elif k == 'Date':
            sum = sum
        else:
            price = checkstockprice(date,k)
            subtotal = price * v
            sum = sum + subtotal
   return sum

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    #print 'orders_file',orders_file
    #df=pd.read_csv(orders_file)
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Symbol','Order','Shares'], header =0)
    #print df
    dfrecords=df.to_dict('records')
    #print dfrecords
    initial_stocklist=[{'Date': start_date, 'CASH':start_val}]
    stocklist = initial_stocklist
    #print stocklist
    listofdates= listworkingdays('AAPL')
    #print listofdates
    workingdates =[]
    for dates in pd.date_range(start=start_date, end=end_date, freq='D'):
        #print dates
        if dates.strftime('%Y-%m-%d') in listofdates:
            workingdates.append(dates.strftime('%Y-%m-%d'))
    #print workingdates
    total=[]
    for i in workingdates:
        today = i
        yesterday = pd.to_datetime(i) + pd.DateOffset(days=-1)
        tomorrow = pd.to_datetime(i) + pd.DateOffset(days=1)
        #print "yesterday",newdate.strftime('%Y-%m-%d')
        #print 'stocklist',stocklist
        for dicts1 in stocklist:
            #print dicts
            if dicts1["Date"] == today:
                newentry = dicts1.copy()

        for dicts in dfrecords:
            #print dicts

            if dicts["Date"] ==i:
                if dicts["Order"] == "BUY":
                   #print dicts["Shares"]
                   #print dicts["Symbol"]
                   stockprice=checkstockprice(today, dicts["Symbol"])
                   #print stockprice
                   newentry["CASH"] = newentry["CASH"] - stockprice * dicts["Shares"]
                   #print newentry["CASH"]
                   tradebookupdate(newentry,dicts["Symbol"],dicts["Shares"])
                   print newentry["CASH"]
                   print stockprice * dicts["Shares"]
                elif dicts["Order"]  == "SELL":
                   #print dicts["Shares"]
                   #print dicts["Symbol"]
                   quantity = -dicts["Shares"]
                   #print quantity
                   stockprice=checkstockprice(today, dicts["Symbol"])
                   newentry["CASH"] = newentry["CASH"] + stockprice * dicts["Shares"]
                   tradebookupdate(newentry,dicts["Symbol"],quantity)
                   print newentry["CASH"]
                   print stockprice * dicts["Shares"]

        newentry["Date"] = tomorrow.strftime('%Y-%m-%d')
        #print "newentry",newentry
        total.append(tradebooktotal(newentry, today))
        #print total
        stocklist.append(newentry.copy())
    #print 'sl', stocklist

    #print pd.Series(workingdates)
    #print pd.Series(total)
    portvals = pd.DataFrame(total, index=workingdates)


    # TODO: Your code here
    #print 'portvals', portvals

    return portvals


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-05'
    end_date = '2011-01-20'
    orders_file = os.path.join("orders", "orders-short.csv")
    start_val = 1000000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    print 'portvals in test_run', portvals
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
