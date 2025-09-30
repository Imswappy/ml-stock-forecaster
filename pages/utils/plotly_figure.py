# pages/utils/plotly_figure.py

import datetime

import dateutil
import pandas as pd
import plotly.graph_objects as go

# Try pandas-ta first; if not available, fall back to 'ta' library.
try:
    import pandas_ta as pta
    _HAVE_PANDAS_TA = True
except Exception:
    _HAVE_PANDAS_TA = False
    # fallback imports from ta library
    from ta.momentum import RSIIndicator
    from ta.trend import MACD

def plotly_table(dataframe):
    headerColor = 'grey'
    rowEvenColor = '#f8fafd'
    rowOddColor = '#e1efff'
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=["<b><b>"]+["<b>"+str(i)[:10]+"<b>" for i in dataframe.columns],
        line_color='#0078ff', fill_color='#0078ff',
        align='center', font=dict(color='white', size=15),height =35,
    ),
    cells=dict(
        values=[["<b>"+str(i)+"<b>" for i in dataframe.index]]+[dataframe[i] for i in dataframe.columns],
        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor]*10],
        align='left', line_color=['white'],font=dict(color=["black"], size=15)
    ))
    ])
    fig.update_layout( height= 400, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def filter_data(dataframe, num_period):
    if num_period == '1mo':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-1)
    elif num_period == '5d':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(days=-5)
    elif num_period == '6mo':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-6)
    elif num_period == '1y':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-1)
    elif num_period == '5y':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-5)
    elif num_period == 'ytd':
        date = datetime.datetime(dataframe.index[-1].year, 1, 1).strftime('%Y-%m-%d')
    else:
        date = dataframe.index[0]
    
    return dataframe.reset_index()[dataframe.reset_index()['Date'] > date]

def sma(series, window=50):
    return series.rolling(window).mean()

def compute_RSI(series, length=14):
    """
    Input: pandas Series of close prices
    Returns: pandas Series of RSI
    """
    if _HAVE_PANDAS_TA:
        # pandas_ta returns a Series (if input is a Series)
        return pta.rsi(series, length=length)
    else:
        return RSIIndicator(series, window=length).rsi()

def compute_MACD(series, fast=12, slow=26, signal=9):
    """
    Returns (macd, macd_signal, macd_diff) as pandas Series
    """
    if _HAVE_PANDAS_TA:
        macd_df = pta.macd(series, fast=fast, slow=slow, signal=signal)
        if isinstance(macd_df, pd.DataFrame) and macd_df.shape[1] >= 3:
            return macd_df.iloc[:, 0], macd_df.iloc[:, 1], macd_df.iloc[:, 2]
        # fallback if pandas_ta returns something else
        return macd_df, None, None
    else:
        macd_obj = MACD(series, window_slow=slow, window_fast=fast, window_sign=signal)
        return macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()

def close_chart(dataframe, num_period=False):
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'],
                        mode='lines',
                        name='Open', line=dict(width=2, color='#5ab7ff')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'],
                        mode='lines',
                        name='Close', line=dict(width=2, color='black')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'],
                        mode='lines', name='High', line=dict(width=2, color='#0078ff')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'],
                        mode='lines', name='Low', line=dict(width=2, color='red')))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#e1efff',
                      legend=dict(yanchor="top", xanchor="right"))
    return fig

def candlestick(dataframe, num_period):
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dataframe['Date'],
                    open=dataframe['Open'], high=dataframe['High'],
                    low=dataframe['Low'], close=dataframe['Close']))
    fig.update_layout(showlegend=False, height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#e1efff')
    return fig

def RSI(dataframe, num_period):
    # compute RSI on the Close series
    dataframe = dataframe.copy()
    dataframe['RSI'] = compute_RSI(dataframe['Close'])
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe.RSI, name='RSI', line=dict(width=2, color='orange'),
    ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[70] * len(dataframe), name='Overbought', line=dict(width=2, color='red', dash='dash'),
    ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[30] * len(dataframe), fill='tonexty', name='Oversold', line=dict(width=2, color='#79da84', dash='dash'),
    ))
    fig.update_layout(yaxis_range=[0, 100],
        height=200, plot_bgcolor='white', paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1))
    return fig

def Moving_average(dataframe, num_period):
    dataframe = dataframe.copy()
    # Using SMA with fallback
    dataframe['SMA_50'] = sma(dataframe['Close'], window=50)
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'],
                        mode='lines',
                        name='Open', line=dict(width=2, color='#5ab7ff')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'],
                        mode='lines',
                        name='Close', line=dict(width=2, color='black')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'],
                        mode='lines', name='High', line=dict(width=2, color='#0078ff')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'],
                        mode='lines', name='Low', line=dict(width=2, color='red')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['SMA_50'],
                        mode='lines', name='SMA 50', line=dict(width=2, color='purple')))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#e1efff',
                      legend=dict(yanchor="top", xanchor="right"))
    return fig

def Moving_average_candle_stick(dataframe, num_period):
    dataframe = dataframe.copy()
    dataframe['SMA_50'] = sma(dataframe['Close'], window=50)
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dataframe['Date'],
                    open=dataframe['Open'], high=dataframe['High'],
                    low=dataframe['Low'], close=dataframe['Close']))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['SMA_50'],
                        mode='lines', name='SMA 50', line=dict(width=2, color='purple')))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#e1efff',
                      legend=dict(yanchor="top", xanchor="right"))
    return fig

def MACD(dataframe, num_period):
    dataframe = dataframe.copy()
    macd_line, macd_signal, macd_hist = compute_MACD(dataframe['Close'])
    # handle None cases gracefully
    dataframe['MACD'] = macd_line
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD'], name='MACD', line=dict(width=2, color='orange'),
    ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD Signal'], name='Signal', line=dict(width=2, color='red', dash='dash'),
    ))
    # optional: add histogram bars or color logic if macd_hist exists
    fig.update_layout(height=200, plot_bgcolor='white', paper_bgcolor='#e1efff',
                      margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1))
    return fig

def Moving_average_forecast(forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast.index[:-30], y=forecast['Close'].iloc[:-30],
                        mode='lines',
                        name='Close Price', line=dict(width=2, color='black')))
    fig.add_trace(go.Scatter(x=forecast.index[-31:], y=forecast['Close'].iloc[-31:],
                        mode='lines', name='Future Close Price', line=dict(width=2, color='red')))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#e1efff',
                      legend=dict(yanchor="top", xanchor="right"))
    return fig
