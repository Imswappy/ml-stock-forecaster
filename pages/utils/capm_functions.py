# pages/utils/capm_functions.py

import numpy as np
import pandas as pd
import plotly.express as px


# Function to plot interactive plot
def interactive_plot(df):
    fig = px.line()
    for i in df.columns[1:]:
         fig.add_scatter(x = df['Date'], y = df[i], name = i)
    fig.update_layout(width = 450,
                      margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# Function to normalize the prices based on the initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i].iloc[0]
    return x

# Function to calculate the daily returns (percentage)
def daily_return(df):
    """
    Expects dataframe with 'Date' and then price columns and 'sp500' column.
    Returns dataframe with same columns where each price column is percent daily return.
    """
    df_daily_return = df.copy().reset_index(drop=True)
    # compute percent change for numeric columns (skip 'Date' if present)
    for col in df_daily_return.columns:
        if col != 'Date':
            # percent change * 100
            df_daily_return[col] = df_daily_return[col].pct_change().fillna(0) * 100
    return df_daily_return

# Function to calculate beta
def calculate_beta(stocks_daily_return, stock):
    # Fit a linear regression (slope = beta, intercept = alpha)
    # stocks_daily_return expected to contain 'sp500' and the stock column
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[stock], 1)
    return b, a
