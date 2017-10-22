"""MC2-P1: Market simulator."""


import numpy as np
import os
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

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
    #print tradebook
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

def listworkingdays(symbol):
    orders_file = os.path.join("data", symbol+".csv")
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Open','High','Low','Close','Volume','Adj Close'], header =0)
    df=df.to_dict('list')
    return df["Date"]



def compute_portvals(start_date, end_date, orders_file, start_val):

    #print 'orders_file',orders_file
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Shares','Symbol','Order'], header =0)
    #print df
    dfrecords=df.to_dict('records')
    #print 'dfrecords',dfrecords
    initial_stocklist=[{'Date': start_date, 'CASH':start_val}]
    stocklist = initial_stocklist
    #print stocklist
    listofdates= listworkingdays('SPY')
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
        #print "yesterday", yesterday
        #print 'stocklist',stocklist
        for dicts1 in stocklist:
            #print dicts1
            if dicts1["Date"] == today:
                newentry = dicts1.copy()

        for dicts in dfrecords:
            #print 'dicts', dicts
            #print i
            if dicts["Date"] ==i:
                if dicts["Order"] == "BUY":
                   #print dicts["Shares"]
                   #print dicts["Symbol"]
                   stockprice=checkstockprice(today, dicts["Symbol"])
                   #print stockprice
                   newentry["CASH"] = newentry["CASH"] - stockprice * dicts["Shares"]
                   #print newentry["CASH"]
                   tradebookupdate(newentry,dicts["Symbol"],dicts["Shares"])
                   #print newentry["CASH"]
                   #print stockprice * dicts["Shares"]
                elif dicts["Order"]  == "SELL":
                   #print dicts["Shares"]
                   #print dicts["Symbol"]
                   quantity = -dicts["Shares"]
                   #print quantity
                   stockprice=checkstockprice(today, dicts["Symbol"])
                   newentry["CASH"] = newentry["CASH"] + stockprice * dicts["Shares"]
                   tradebookupdate(newentry,dicts["Symbol"],quantity)
                   #print newentry["CASH"]
                   #print stockprice * dicts["Shares"]

        newentry["Date"] = tomorrow.strftime('%Y-%m-%d')
        #print "newentry",newentry
        total.append(tradebooktotal(newentry, today))
        #print total
        stocklist.append(newentry.copy())
    #print 'sl', stocklist
    #print pd.Series(workingdates)
    #print pd.Series(total)
    portvals = pd.DataFrame(total, index=workingdates, columns=['Portfolio'])
    # TODO: Your code here
    #print 'portvals', portvals

    return portvals


##############################
#below is from the analysis.py
##############################
def get_portfolio_value(prices, allocs, start_val=1):

    # TODO: Your code here
    port = prices.copy()
    for col in port:
        port[col]/=port[col].iloc[0]
    #print "port",port
    daily_port_val = port.dot(allocs)
    #print "daily port value", daily_port_val
    port_val = daily_port_val * start_val
    return port_val

def get_original_portfolio_value(prices, allocs):

    # TODO: Your code here
    port = prices.copy()
    #for col in port:
    #    port[col]/=port[col].iloc[0]
    #print "port",port
    daily_port_val = port.dot(allocs)
    #print "daily port value", daily_port_val
    port_val = daily_port_val
    return port_val

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    # TODO: Your code here
    df=port_val.copy()
    daily_ret = (df/ df.shift(1))-1
    daily_ret.fillna(0)
    cum_ret=(df[-1]-df[0])/df[0]

    #print "cum_ret",cum_ret
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    #print std_daily_ret
    #print avg_daily_ret
    sharpe_ratio=((252)**0.5)*avg_daily_ret/std_daily_ret
    #print sharpe_ratio
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def get_portfolio_bollinger_bands(port_val, daily_rf=0, samples_per_year=252):

    # TODO: Your code here
    df=port_val.copy()
    sma=pd.rolling_mean(df, 20)
    std_sma = pd.rolling_std(df,20)
    upper_band= sma+ 2* std_sma
    lower_band= sma- 2* std_sma

    return sma, upper_band, lower_band

def detect_events(port_val, sma, upper_band, lower_band):
    df = port_val.copy()
    events=list(xrange(len(df.index)))

    for i in range(0,len(df.index)):
        events[i]=0
        if (i>1) and (df[i-1]< lower_band[i-1]) and (df[i]> lower_band[i]) :
            events[i]='longentry'
        if (i>1) and (df[i-1]< sma[i-1]) and (df[i]> sma[i]) :
            events[i]='longexit'
        if (i>1) and (df[i-1]> upper_band[i-1]) and (df[i]< upper_band[i]) :
            events[i]='shortentry'
        if (i>1) and (df[i-1]> sma[i-1]) and (df[i]< sma[i]) :
            events[i]= 'shortexit'
    print events
    return events


def orderbook_generator(events, symbol, stock_shares=100 ):
    df= events.copy()
    #print df[df['events'].isin(['longentry', 'longexit','shortentry','shortexit'])]
    columns = ['Date','Shares', 'Symbol', 'Order']
    order_book = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
    columns2 = ['events',]
    updated_events = pd.DataFrame(data=np.zeros((0,len(columns2))), columns=columns2)
    #print 'order_book', order_book
    flag_long =0
    flag_short =0
    for i in range(0,len(df.index)):
        if df['events'][i] =='longentry':
            if flag_long == 0:
                flag_long = 1
                order_book = order_book.append({'Date':df.index[i], 'Symbol':symbol, 'Order':'BUY','Shares':stock_shares},ignore_index=True)
                updated_events = updated_events.append({'Date': df.index[i], 'events':'longentry'},ignore_index=True)
            else:
                flag_long = 1
        if df['events'][i] =='longexit':
            if flag_long == 1:
                flag_long = 0
                order_book = order_book.append({'Date':df.index[i], 'Symbol':symbol, 'Order':'SELL','Shares':stock_shares},ignore_index=True)
                updated_events = updated_events.append({'Date': df.index[i], 'events':'longexit'},ignore_index=True)
            else:
                flag_long =0
        if df['events'][i] =='shortentry':
            if flag_short ==0:
                flag_short =1
                order_book = order_book.append({'Date':df.index[i], 'Symbol':symbol, 'Order':'SELL','Shares':stock_shares},ignore_index=True)
                updated_events = updated_events.append({'Date': df.index[i], 'events':'shortentry'},ignore_index=True)
            else:
                flag_short =1
        if df['events'][i] =='shortexit':
            if flag_short ==1:
                flag_short =0
                order_book = order_book.append({'Date':df.index[i], 'Symbol':symbol, 'Order':'BUY','Shares':stock_shares},ignore_index=True)
                updated_events = updated_events.append({'Date': df.index[i], 'events':'shortexit'},ignore_index=True)
            else:
                flag_short =0
    order_book1 = order_book.set_index(['Date'])
    #print 'order_book', order_book1
    order_book1.to_csv('orders.csv')
    #print 'updated_events', updated_events
    updated_events1 = updated_events.set_index(['Date'])
    #updated_events1.to_csv('events.csv')
    return updated_events1

def plot_normalized_data(df, name, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    #TODO: Your code here
    df2=df.copy()
    for col in df2:
        df2[col]/=df2[col].iloc[0]
    #print df2
    ax=df2.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot()
    fig=plt.gcf()

    fig.savefig(name)

def assess_portfolio(start_date, end_date, symbols, allocs, number_of_stocks=1):
    """Simulate and assess the performance of a stock portfolio."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    # Get daily portfolio value
    port_val = get_original_portfolio_value(prices, allocs)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)
    sma, upper_band, lower_band = get_portfolio_bollinger_bands(port_val)
    events = detect_events(port_val, sma, upper_band, lower_band)

    # Compare daily portfolio value with SPY using a normalized plot
    df_bband = pd.concat([prices, sma, upper_band, lower_band], axis=1)
    df_bband.columns =['Price','Sma', 'Upperband', 'Lowerband']

    df_events=pd.DataFrame(events, index=df_bband.index)
    df_events.rename(columns = {list(df_events)[0]: 'events'}, inplace = True)
    #print 'df_events', df_events
    #given bollinger band strategy generate order book
    updated_events = orderbook_generator(df_events, ''.join(symbols),number_of_stocks)

    return df_bband, updated_events
##########################
#above is from analysis.py
##########################


def plot_bollinger_band_data(df_bband, symbol, df_events, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    df=df_bband.copy()
    dfevents= df_events.copy()
    #dfevents.to_csv('plot_bollinger_band.csv')
    ax = plt.gca()
    #df.plot(ax = ax)
    #ax = df.plot(title=title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    df['Price'].plot(label=symbol, ax=ax,color='b',figsize=(16,10), legend=True)
    df['Sma'].plot(label='SMA', ax=ax,color='y', legend=True)
    df['Upperband'].plot(label='Bollinger Bands', ax=ax,color='cyan', legend=True)
    df['Lowerband'].plot(ax=ax,color='cyan', legend=False)



    longentry = dfevents[dfevents['events'] =='longentry']

    longexit = dfevents[dfevents['events'] =='longexit']
    shortentry = dfevents[dfevents['events'] =='shortentry']
    shortexit = dfevents[dfevents['events'] =='shortexit']
    #print longentry.index
    plt.vlines(longentry.index, plt.ylim()[0], plt.ylim()[1] ,color ='g',linewidth=1)
    plt.vlines(longexit.index, plt.ylim()[0], plt.ylim()[1],color ='k')
    plt.vlines(shortentry.index, plt.ylim()[0], plt.ylim()[1],color ='r')
    plt.vlines(shortexit.index, plt.ylim()[0], plt.ylim()[1],color ='k')


    fig=plt.gcf()
    fig.savefig('bband.png')
###############################
#above is from util.py
###############################

def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2007-12-31'       #start_date
    end_date = '2009-12-31'         #end_date
    symbol ='IBM'   #list of symbols
    start_val = 10000
    number_of_stocks = 100
    symbol_allocations = OrderedDict([(symbol, 1)])
    symbols= symbol_allocations.keys()
    allocs = symbol_allocations.values()

    # Simulate a SPX-only reference portfolio to get stats
    prices_SPY= get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPY = prices_SPY[['$SPX']]  # remove SPY

    portvals_SPY = get_portfolio_value(prices_SPY, [1.0])
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(portvals_SPY) #calculate SPY cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

    # Process orders
    #optimize_portfolio(start_date,end_date,symbols)

    #given start_date, end_date, symbol, start_val, calculate bollinger band's sma, upper band, lower band and save in df_bband
    # based on sma, upper band, lower band, generate bollinger band strategy events and save in df_events
    df_bband, updated_events= assess_portfolio(start_date, end_date, symbols, allocs, number_of_stocks)

    plot_bollinger_band_data(df_bband,symbol, updated_events, title="Bollinger Bands")
    #print 'start compute portvals'
    portvals = compute_portvals(start_date, end_date, "orders.csv", start_val)
    #portvals.to_csv('portvals.csv')
    portvals['SPY'] = prices_SPY #add SPY to the portvals
    plot_normalized_data(portvals, 'Daily portfolio value.png', title="Daily portfolio value")

    ##########below can be commented out
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    #print 'portvals in test_run', portvals
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY: {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY: {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value

    #df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    #plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
