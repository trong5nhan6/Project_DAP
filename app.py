import streamlit as st
import streamlit.components.v1 as components
import os
import glob
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

# ===== Cấu hình trang =====
st.set_page_config(layout="wide", page_title="Stock Dashboard")

# ===== CSS tùy chỉnh =====
st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem 0rem 2rem !important; }
        .st-emotion-cache-ue6h4q,
        div[data-testid="stHorizontalBlock"] {
            background-color: transparent !important;
            box-shadow: none !important;
            padding: 0 !important; margin: 0 !important;
        }
        .nav-link-selected {
            background-color: #0072E0 !important;
            color: white !important;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Tiêu đề =====
st.markdown("<h1 style='text-align: center; color: white;'>Stock Data Dashboard</h1>", unsafe_allow_html=True)

# ===== Layout chính =====
left_col, right_col = st.columns([3, 1], gap="large")

with left_col:
    # ==== Chọn mã chứng khoán ====
    selected = option_menu(
        menu_title=None,
        options=['GG', 'Netflix', 'Apple', 'Nvidia', 'Tesla'],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#bbb", "font-size": "16px"},
            "nav-link": {
                "font-size": "16px", "text-align": "center",
                "margin": "0px", "color": "#ddd", "--hover-color": "#444",
            },
            "nav-link-selected": {
                "background-color": "#0072E0",
                "color": "#fff", "font-weight": "bold"
            },
        }
    )

    # ==== Chọn mô hình ====
    st.subheader(f"🔎 Model dự đoán - {selected}")
    model_type = st.radio(
        "Chọn mô hình dự đoán:",
        ["Linear", "RandomForest", "LSTM"],
        horizontal=True
    )

    # ==== Load dữ liệu ====
    data_path = f"data/{selected.lower()}.csv"
    checkpoint_path = f"model/checkpoint/{selected.lower()}"

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        today = df['Date'].max().date()

        future_date = st.date_input("📅 Chọn ngày muốn dự đoán:", min_value=today + datetime.timedelta(days=1))
        n_steps = (future_date - today).days

        try:
            # Load model và scaler
            if model_type in ['Linear', 'RandomForest']:
                model = joblib.load(os.path.join(checkpoint_path, f"{model_type}.pkl"))
            else:
                model = load_model(os.path.join(checkpoint_path, f"{model_type}.h5"), compile=False)
            scaler = joblib.load(os.path.join(checkpoint_path, "scaler.pkl"))

            # ==== Hàm dự đoán ====
            def recursive_predict(model, recent_series, n_steps, window_size, model_type, scaler=None):
                # Scale recent_series nếu scaler có
                if scaler:
                    series = scaler.transform(np.array(recent_series).reshape(-1, 1)).flatten().tolist()
                else:
                    series = list(recent_series[-window_size:])

                preds = []
                for _ in range(n_steps):
                    x_input = np.array(series[-window_size:])
                    if model_type in ["LSTM", "GRU"]:
                        x_input = x_input.reshape(1, window_size, 1)
                        pred = model.predict(x_input, verbose=0)
                        pred_value = float(pred[0][0])  # 💥 đảm bảo là số float
                    else:
                        x_input = x_input.reshape(1, -1)
                        pred_value = float(model.predict(x_input)[0])  # 💥 đảm bảo là số float
                    preds.append(pred_value)
                    series.append(pred_value)

                if scaler:
                    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                return preds


            close_series = df['Close'].values
            preds = recursive_predict(model, close_series, n_steps=n_steps, window_size=30, model_type=model_type, scaler=scaler)

            # ==== Hiển thị kết quả ====
            future_dates = pd.date_range(start=today + datetime.timedelta(days=1), periods=n_steps)
            df_forecast = pd.DataFrame({'Date': future_dates, 'Predicted': preds})

            st.success(f"💰 Giá dự đoán vào {future_date.strftime('%Y-%m-%d')} bằng mô hình {model_type}: **${preds[-1]:.2f}**")

            # ==== Vẽ biểu đồ ====
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_forecast['Date'], y=df_forecast['Predicted'],
                mode='lines+markers',
                name='Giá dự đoán',
                line=dict(color='orange', dash='dot')
            ))
            fig.update_layout(
                title=f"Dự đoán giá cổ phiếu {selected} ({model_type})",
                xaxis_title="Ngày", yaxis_title="Giá (USD)",
                template='plotly_dark', height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Lỗi dự đoán: {e}")

    else:
        st.warning("⚠️ Không tìm thấy dữ liệu cho mã này.")

    # ==== Biểu đồ HTML từ thư mục riêng ====
    st.subheader(f"📊 Data Visualized - {selected}")
    stock_dir = f"figures/{selected.lower()}"

    if os.path.isdir(stock_dir):
        html_files = sorted(glob.glob(os.path.join(stock_dir, "figure_*.html")))
        for html_path in html_files:
            note_path = html_path.replace("figure_", "note_").replace(".html", ".txt")
            if os.path.exists(note_path):
                with open(note_path, "r", encoding="utf-8") as nf:
                    st.markdown(nf.read())
            with open(html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=500, scrolling=False)
    else:
        st.warning(f"⚠️ Không tìm thấy thư mục `{stock_dir}/` chứa dữ liệu của mã này.")

# ==== ChatBot placeholder ====
with right_col:
    st.subheader("ChatBot")
    st.write("💬 Khu vực này để bạn tích hợp chatbot xử lý hội thoại.")
    st.text_area("Bot output sẽ hiển thị ở đây...", height=250)
    st.text_input("Nhập câu hỏi:")
