import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("📊 Sales Forecasting Dashboard")

# Upload file
uploaded_file = st.file_uploader("forcasting.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Data preparation
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["total_sales"] = df["price"] * df["quantity"]

        df = df.dropna(subset=["date"])

        # Aggregate daily sales
        daily_sales = df.groupby("date")["total_sales"].sum()
        daily_sales = daily_sales.sort_index()
        daily_sales = daily_sales.asfreq("D").fillna(0)

        # Plot daily sales
        st.subheader("📈 Daily Sales Trend")
        fig, ax = plt.subplots()
        daily_sales.plot(ax=ax)
        ax.set_title("Daily Sales")
        st.pyplot(fig)

        # Forecast settings
        st.subheader("⚙️ Forecast Settings")
        steps = st.slider("Forecast Days", 7, 60, 14)

        # Train model
        model = SARIMAX(
            daily_sales,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(disp=False)

        # Forecast
        forecast = results.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Plot forecast
        st.subheader("🔮 Forecast")
        fig2, ax2 = plt.subplots()

        daily_sales.plot(ax=ax2, label="Historical")
        forecast_mean.plot(ax=ax2, label="Forecast")

        ax2.fill_between(
            forecast_ci.index,
            forecast_ci.iloc[:, 0],
            forecast_ci.iloc[:, 1],
            alpha=0.3
        )

        ax2.legend()
        st.pyplot(fig2)

        # Show forecast values
        st.subheader("📋 Forecast Data")
        forecast_df = pd.DataFrame({
            "Forecast": forecast_mean,
            "Lower CI": forecast_ci.iloc[:, 0],
            "Upper CI": forecast_ci.iloc[:, 1]
        })

        st.write(forecast_df)

    except Exception as e:
        st.error(f"Error: {e}")