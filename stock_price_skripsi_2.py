import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title of the web app
st.title('Prediksi Harga Saham PT Telkomsel (TLKM)')

# Tampilkan deskripsi aplikasi
st.write("""
Aplikasi ini memungkinkan Anda untuk memprediksi harga saham PT Telkomsel (TLKM.JK) menggunakan model LSTM yang telah dilatih.
""")

# Fungsi untuk mengambil data saham Telkomsel menggunakan yfinance
def get_stock_data():
    ticker = "TLKM.JK"
    data = yf.download(ticker, period="1mo")  # Menyesuaikan periode data
    return data

# Fungsi untuk menampilkan data saham Telkomsel
data = get_stock_data()
st.write("Data Saham PT Telkomsel (TLKM.JK) Terakhir:", data.tail())

# Preprocessing dan normalisasi data
def preprocess_data(data):
    data = data[['Adj Close']]  # Menggunakan harga penutupan yang disesuaikan
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

data_scaled, scaler = preprocess_data(data)

# Fungsi untuk membuat sliding window
def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 10  # Jumlah hari sebelumnya untuk prediksi
X, y = create_sliding_window(data_scaled, window_size)

# Load model LSTM yang telah dilatih
model = load_model('lstm_model.h5')

# Fungsi untuk memprediksi harga saham
def predict_future_price(model, last_known_data, scaler, days_ahead):
    predicted_prices = []
    current_data = last_known_data.reshape(1, window_size, 1)

    for _ in range(days_ahead):
        next_price = model.predict(current_data)
        predicted_prices.append(next_price[0])
        current_data = np.append(current_data[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices

# Input untuk jumlah hari ke depan yang ingin diprediksi
days_ahead = st.slider('Jumlah Hari untuk Prediksi', min_value=1, max_value=30, value=7)

# Ambil data terakhir untuk prediksi
last_known_data = data_scaled[-window_size:]

# Prediksi harga saham
predicted_prices = predict_future_price(model, last_known_data, scaler, days_ahead)

# Tampilkan hasil prediksi
st.write("Hasil Prediksi Harga Saham untuk 30 Hari ke Depan:")
for i, price in enumerate(predicted_prices, start=1):
    st.write(f"Hari ke-{i}: Rp {price[0]:,.2f}")

# Visualisasi hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(range(1, days_ahead + 1), predicted_prices, label='Predicted Prices', marker='o', linestyle='--')
plt.title('Prediksi Harga Saham untuk 30 Hari ke Depan')
plt.xlabel('Hari ke-')
plt.ylabel('Harga Saham (Rp)')
plt.legend()
plt.grid()
st.pyplot()

# Plot harga saham historis untuk perbandingan
st.write("Grafik Harga Saham Terakhir PT Telkomsel (TLKM.JK):")
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Adj Close'], label='Harga Saham Historis', color='blue')
plt.title('Harga Saham Terakhir PT Telkomsel')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham (Rp)')
plt.legend()
st.pyplot()
