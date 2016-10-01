
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret
    #plot_normalized_data([sma, upper_band, lower_band], title="Daily portfolio value and SPY")



    # Simulate a SPX-only reference portfolio to get stats
    prices_SPY= get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPY = prices_SPY[['SPY']]  # remove SPY

    portvals_SPY = get_portfolio_value(prices_SPY, [1.0])




def plot_normalized_data2(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
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
    #dfbar2=dfbar.copy()
    df2.fillna(method='bfill', limit=31)
    #print 'df2 is',df2
    #for col in df2:
   #     df2[col]/=df2[col].iloc[0]
    #print df2
    rb = df.loc[:,['PRICE','SMA', 'UPPERBAND', 'LOWERBAND']]
    #print rb
    ax=rb.plot(title=title, fontsize=12)
    #ax.set_xlabel(xlabel)
    #ax.set_ylabel(ylabel)
    entry = df.loc[:,['events']]#,'longexit']]
    #print entry['events']
    #ax=plt.gca()
    #ax = df2.plot(label='longentry')
    ymin, ymax = ax.get_ylim()
    print ymin
    print ymax
    #plt.axvline(longentry, ymin, ymax)
    #ax= df2.plot()#label=symbol)
    #ymin, ymax = ax.get_ylim()
    #fig=plt.gcf()
    """
    ax=plt.gca()
    #df2.plot(ax=ax)
    for i in range(0,len(entry.index)):
        #print entry['events'][i]
        if entry['events'][i] =='shortexit':
            #print entry[i]
            print entry['events'][i].index
            #dates=entry[i]
            ax.axvline(x=entry[i].index, ymin =ymin, ymax=ymax)#x=trading_day, color='g')
    plt.show()
    fig.savefig('new.png')
    """

def plot_bollinger_band_data_direct(df_bband,  title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    df=df_bband.copy()

    #dfevents.to_csv('plot_bollinger_band.csv')
    #ax = plt.gca()
    #df.plot(ax = ax)
    ax = df.plot(title=title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


    fig=plt.gcf()
    fig.savefig('bband1.png')