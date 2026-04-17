import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title('Stock Forecast App')

# Dropdown for stock selection
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'INTC', 'NOK')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider for years of prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

if selected_stock=="GOOG":
    data=pd.read_csv('GOOG.csv')
    st.subheader('Raw data')
    st.write(data.tail())
elif selected_stock=="AAPL":
    data = pd.read_csv('AAPL(1).csv')
    st.subheader('Raw data')
    st.write(data.tail())
elif selected_stock=="MSFT":
    data = pd.read_csv('MSFT1.csv')
    st.subheader('Raw data')
    st.write(data.tail())
elif selected_stock=="GME":
    data = pd.read_csv('GME1.csv')
    st.subheader('Raw data')
    st.write(data.tail())
elif selected_stock=="INTC":
    data = pd.read_csv('INTC.csv')
    st.subheader('Raw data')
    st.write(data.tail())
elif selected_stock=="NOk":
    data = pd.read_csv('NOK.csv')
    st.subheader('Raw data')
    st.write(data.tail())


# Plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)

# Prepare data for forecasting
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Forecasting with Prophet
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
