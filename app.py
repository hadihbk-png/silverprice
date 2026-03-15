import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Silver Price Predictor", layout="wide")
st.title("Silver Price Analysis and Forecast (SI=F)")

@st.cache_data
def get_data():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5*365 + 10) # 5 years
    
    import requests
    session = requests.Session()
    session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    data = yf.download("SI=F", start=start_date, end=end_date, session=session)
    
    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'].copy()
        if isinstance(df, pd.Series):
            df = df.to_frame(name='Close')
        else:
            df.columns = ['Close']
    else:
        df = data[['Close']].copy()
    
    df.dropna(inplace=True)
    return df

df = get_data()

if df.empty:
    st.error("Failed to fetch data from Yahoo Finance. This sometimes happens on cloud servers due to rate limits. Please try again later.")
    st.stop()

st.subheader("Historical Data (Last 5 Years)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index, y=df['Close'].values.flatten(), mode='lines', name='Close Price', line=dict(color='#00b4d8')))
fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Model Evaluation & Optimization")
st.markdown("We automatically test several models (Exponential Smoothing variants and ARIMA configurations) and select the one that generates the **lowest Root Mean Squared Error (RMSE)** on the test dataset.")

train_size = int(len(df) * 0.95)
train, test = df.iloc[:train_size], df.iloc[train_size:]
train_vals = np.asarray(train['Close'].values.flatten())
test_vals = np.asarray(test['Close'].values.flatten())

best_rmse = float('inf')
best_model_name = ""
best_predictions = None

with st.spinner("Finding the model with the lowest RMSE..."):
    # Model 1: Exponential Smoothing Additive
    try:
        m1 = ExponentialSmoothing(train_vals, trend='add', seasonal=None, initialization_method="estimated").fit()
        p1 = m1.forecast(len(test))
        rmse1 = np.sqrt(mean_squared_error(test_vals, p1))
        if rmse1 < best_rmse:
            best_rmse = rmse1
            best_model_name = "Exponential Smoothing (Additive)"
            best_predictions = p1
    except: pass

    # Model 2: Exponential Smoothing Damped
    try:
        m2 = ExponentialSmoothing(train_vals, trend='add', seasonal=None, damped_trend=True, initialization_method="estimated").fit()
        p2 = m2.forecast(len(test))
        rmse2 = np.sqrt(mean_squared_error(test_vals, p2))
        if rmse2 < best_rmse:
            best_rmse = rmse2
            best_model_name = "Exponential Smoothing (Damped)"
            best_predictions = p2
    except: pass

    # Model 3: ARIMA (5,1,0)
    try:
        m3 = ARIMA(train_vals, order=(5,1,0)).fit()
        p3 = m3.forecast(steps=len(test))
        rmse3 = np.sqrt(mean_squared_error(test_vals, p3))
        if rmse3 < best_rmse:
            best_rmse = rmse3
            best_model_name = "ARIMA (5,1,0)"
            best_predictions = p3
    except: pass

    # Model 4: ARIMA (1,1,1)
    try:
        m4 = ARIMA(train_vals, order=(1,1,1)).fit()
        p4 = m4.forecast(steps=len(test))
        rmse4 = np.sqrt(mean_squared_error(test_vals, p4))
        if rmse4 < best_rmse:
            best_rmse = rmse4
            best_model_name = "ARIMA (1,1,1)"
            best_predictions = p4
    except: pass

st.success(f"🏆 **Best Model Found:** {best_model_name} with RMSE = **{best_rmse:.4f}**")

fig_eval = go.Figure()
fig_eval.add_trace(go.Scatter(x=test.index, y=test_vals, mode='lines', name='Actual Test Data', line=dict(color='#00b4d8')))
fig_eval.add_trace(go.Scatter(x=test.index, y=best_predictions, mode='lines', name=f'{best_model_name} Predictions', line=dict(dash='dash', color='#ef233c')))
fig_eval.update_layout(title="Test Set Predictions vs Actuals", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.plotly_chart(fig_eval, use_container_width=True)

st.subheader("1-Year Forecast")
with st.spinner(f"Forecasting next year using {best_model_name}..."):
    full_vals = np.asarray(df['Close'].values.flatten())
    forecast_steps = 252

    if "Exponential" in best_model_name:
        is_damped = "Damped" in best_model_name
        final_m = ExponentialSmoothing(full_vals, trend='add', seasonal=None, damped_trend=is_damped, initialization_method="estimated").fit()
        forecast = final_m.forecast(forecast_steps)
    else:
        order = (5,1,0) if "5,1,0" in best_model_name else (1,1,1)
        final_m = ARIMA(full_vals, order=order).fit()
        forecast = final_m.forecast(steps=forecast_steps)

last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1), periods=forecast_steps)

fig_forecast = go.Figure()
# Show only last year of historical data for better zoom
recent_df = df.iloc[-252:]
fig_forecast.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'].values.flatten(), mode='lines', name='Recent Historical', line=dict(color='#00b4d8')))
fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecasted Future', line=dict(color='#ff9f1c')))
fig_forecast.update_layout(title="Silver Price 1-Year Forecast (Zoomed In)", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.plotly_chart(fig_forecast, use_container_width=True)
