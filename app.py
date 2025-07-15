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


# ===== Cấu hình trang =====
st.set_page_config(layout="wide", page_title="Stock Dashboard")

# ===== CSS tùy chỉnh =====
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        .st-emotion-cache-ue6h4q {
            background-color: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stHorizontalBlock"] {
            background-color: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
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
    # ==== Menu chọn mã chứng khoán ====
    selected = option_menu(
        menu_title=None,
        options=['GG', 'Netflix', 'Apple', 'Nvidia', 'Tesla'],
        icons=['bar-chart', 'film', 'phone', 'box', 'car'],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#bbb", "font-size": "16px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "color": "#ddd",
                "--hover-color": "#444",
            },
            "nav-link-selected": {
                "background-color": "#0072E0",
                "color": "#fff",
                "font-weight": "bold"
            },
        }
    )

    # ==== Model dự đoán ====
    st.subheader(f"Model dự đoán - {selected}")
    st.write(f"Hiển thị kết quả dự đoán riêng cho mã **{selected}**.")
    # === Load data và model ===
    data_path = f"data/{selected.lower()}.csv"
    model_path = f"model/checkpoint/{selected.lower()}_model.pkl"

    if os.path.exists(data_path) and os.path.exists(model_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        model = joblib.load(model_path)

        today = df['Date'].max().date()
        future_date = st.date_input("📅 Chọn ngày muốn dự đoán:", min_value=today + datetime.timedelta(days=1))
        
        # === Dự đoán nhiều bước bằng recursive_predict ===
        def recursive_predict(model, recent_series, n_steps=1, window_size=30):
            preds = []
            series = list(recent_series[-window_size:])
            for _ in range(n_steps):
                x_input = np.array(series[-window_size:]).reshape(1, -1)
                pred = model.predict(x_input)[0]
                preds.append(pred)
                series.append(pred)
            return preds

        n_steps = (future_date - today).days
        close_series = df['Close'].values

        if n_steps > 0:
            try:
                preds = recursive_predict(model, close_series, n_steps=n_steps)
                predicted_price = preds[-1]
                st.success(f"💰 Giá dự đoán vào {future_date.strftime('%Y-%m-%d')}: **${predicted_price:.2f}**")
                # Vẽ biểu đồ thực tế và dự đoán

                # Tạo mảng ngày tương lai dựa vào số bước n_steps
                future_dates = pd.date_range(start=today + datetime.timedelta(days=1), periods=n_steps)

                # DataFrame kết quả dự đoán
                df_forecast = pd.DataFrame({'Date': future_dates, 'Predicted': preds})

                # Tạo biểu đồ
                fig = go.Figure()

                # Dự đoán
                fig.add_trace(go.Scatter(
                    x=df_forecast['Date'], y=df_forecast['Predicted'],
                    mode='lines+markers',
                    name='Giá dự đoán',
                    line=dict(color='orange', dash='dot')
                ))

                fig.update_layout(
                    title=f"Dự đoán giá cổ phiếu {selected}",
                    xaxis_title="Ngày",
                    yaxis_title="Giá (USD)",
                    template='plotly_dark',
                    height=500
                )

                # Hiển thị biểu đồ
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi dự đoán: {e}")
    else:
        st.warning("Không tìm thấy dữ liệu hoặc mô hình cho mã này.")


    # ==== Biểu đồ HTML từ thư mục riêng ====
    st.subheader(f"Data Visualized - {selected}")
    stock_dir = f"figures/{selected.lower()}"

    if os.path.isdir(stock_dir):
        html_files = sorted(glob.glob(os.path.join(stock_dir, "figure_*.html")))

        if html_files:
            for html_path in html_files:
                # Dò tên ghi chú tương ứng (note_#.txt)
                note_path = html_path.replace("figure_", "note_").replace(".html", ".txt")

                if os.path.exists(note_path):
                    with open(note_path, "r", encoding="utf-8") as nf:
                        note_text = nf.read()
                    st.markdown(f"{note_text}")
                else:
                    st.markdown("_Không có ghi chú cho biểu đồ này._")

                # Hiển thị biểu đồ
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=500, scrolling=False)

                # components.html(html_content, height=500, scrolling=True)
        else:
            st.warning(f"Không tìm thấy file `figure_*.html` trong thư mục `{stock_dir}/`.")
    else:
        st.warning(f"Không tìm thấy thư mục `{stock_dir}/` chứa dữ liệu của mã này.")

with right_col:
    st.subheader("ChatBot")
    st.write("Khu vực này để bạn tích hợp chatbot xử lý hội thoại.")
    st.text_area("Bot output sẽ hiển thị ở đây...", height=250)
    st.text_input("Nhập câu hỏi:")
