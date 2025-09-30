# pages/utils/__init__.py

# Utility package for the Stock Forecasting app.
# Export commonly-used helpers if you want:
from .capm_functions import (calculate_beta, daily_return, interactive_plot,
                             normalize)
from .model_train import (evaluate_model, get_data, get_differencing_order,
                          get_forecast, get_rolling_mean, inverse_scaling,
                          scaling)
from .plotly_figure import (MACD, RSI, Moving_average,
                            Moving_average_candle_stick,
                            Moving_average_forecast, candlestick, close_chart,
                            filter_data, plotly_table)
