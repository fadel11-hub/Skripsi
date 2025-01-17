import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf


# Fungsi untuk memuat model dan scaler
def load_model_and_scaler():
    # Ganti dengan path model LSTM Anda
    model = load_model("lstm_model.h5")
    # Pastikan scaler cocok dengan model Anda
    scaler = MinMaxScaler(feature_range=(0, 1))
    return model, scaler


# Fungsi untuk memprediksi harga saham di masa depan
def predict_future_prices(model, scaler, data, window_size, days_ahead):
    # Normalisasi data
    scaled_data = scaler.transform(data)
    last_known_data = scaled_data[-window_size:]

    # Prediksi hari ke depan
    predictions = []
    for _ in range(days_ahead):
        input_data = last_known_data.reshape(1, window_size, 1)
        predicted_price = model.predict(input_data)[0, 0]
        predictions.append(predicted_price)
        last_known_data = np.append(last_known_data[1:], predicted_price)

    # Invers transform ke skala asli
    predictions_original = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))
    return predictions_original.flatten()


# Fungsi untuk memuat data dari Yahoo Finance
def load_data(ticker):
    START = "2015-01-01"
    TODAY = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # Memastikan kolom Date tersedia
    return data


# Fungsi utama Streamlit
def main():
    st.title("Prediksi Harga Saham PT Telkom Indonesia")
    st.write("Aplikasi ini memprediksi harga saham menggunakan model LSTM.")

    # Pilihan saham
    stocks = ("AAPL", "GOOG", "MSFT", "GME", "BBCA.JK", "TLKM.JK", "GOTO.JK")
    selected_stock = st.selectbox("Pilih dataset untuk prediksi", stocks)

    # Load data
    data_load_state = st.text("Loading data...")
    data = load_data(selected_stock)
    data_load_state.text("Loading data... done!")

    # Tampilkan kolom yang tersedia
    st.subheader(f"Kolom yang tersedia dalam {selected_stock}:")
    st.write(data.columns.tolist())

    # Pilih kolom untuk prediksi
    if "Adj Close" in data.columns:
        column = "Adj Close"
    elif "Close" in data.columns:
        column = "Close"
    else:
        st.error("Kolom 'Adj Close' atau 'Close' tidak tersedia dalam dataset.")
        return

    st.subheader(f"Data {selected_stock} (menggunakan kolom '{column}')")
    st.write(data.head())

    # Konversi kolom Date ke datetime dan set sebagai indeks
    if 'Date' not in data.columns:
        st.error("Dataset tidak memiliki kolom 'Date'.")
        return
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Plot data historis
    st.subheader("Grafik Harga Saham Historis")
    plt.figure(figsize=(12, 6))
    plt.plot(data[column], label="Harga Saham Historis")
    plt.legend()
    st.pyplot(plt)

    # Memuat model dan scaler
    model, scaler = load_model_and_scaler()
    scaler.fit(data[[column]])

    # Input parameter prediksi
    days_ahead = st.number_input("Jumlah hari ke depan untuk prediksi:", min_value=1, max_value=365, value=7)
    window_size = 10
    if len(data) < window_size:
        st.error(f"Data tidak cukup untuk prediksi. Minimal {window_size} data diperlukan.")
        return

    # Tombol untuk prediksi
    if st.button("Prediksi Harga Saham"):
        predictions = predict_future_prices(model, scaler, data[[column]].values, window_size, days_ahead)

        # Tampilkan hasil prediksi
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
        prediction_df = pd.DataFrame({"Tanggal": future_dates, "Harga Prediksi": predictions})
        st.subheader("Hasil Prediksi")
        st.write(prediction_df)

        # Plot hasil prediksi
        st.subheader("Grafik Prediksi Harga Saham")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[column], label="Harga Historis")
        plt.plot(future_dates, predictions, label="Harga Prediksi", linestyle="dashed", color="orange")
        plt.legend()
        st.pyplot(plt)


if __name__ == "__main__":
    main()
