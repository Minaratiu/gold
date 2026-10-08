"""
Ứng dụng web dự đoán giá vàng bằng LSTM (Streamlit, một page, hai tab).

Luồng xử lý chung cho CẢ HAI tab (chỉ khác nguồn dữ liệu):
    Dữ liệu -> validate -> lấy Close -> sắp xếp Date -> lấy TIMESTEP giá gần nhất
    -> scaler.transform() -> reshape (1, TIMESTEP, 1) -> LSTM -> scaler.inverse_transform()
    -> tính chênh lệch / % / xu hướng -> vẽ biểu đồ

Ghi chú kỹ thuật (không hiển thị ra giao diện):
- TIMESTEP được đọc trực tiếp từ input_shape của model (không tự đoán).
- scaler.pkl được nạp bằng joblib.load(); CHỈ dùng transform / inverse_transform, không fit lại.
- App không train, không fine-tune, không ghi đè model hay scaler.
"""
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ======================= CẤU HÌNH =======================
BASE = Path(__file__).parent
MODEL_PATH = BASE / "gold_price_lstm_model.h5"
SCALER_PATH = BASE / "scaler.pkl"
CSV_PATH = BASE / "goldstock.csv"
TICKER = "GC=F"          # Gold Futures trên Yahoo Finance, chu kỳ 1 ngày

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gold_app")

st.set_page_config(page_title="Dự đoán giá vàng LSTM", layout="wide",
                   initial_sidebar_state="collapsed")

# ======================= GIAO DIỆN (CSS) =======================
st.markdown(
    """
    <style>
    /* Khung nội dung: giới hạn bề rộng để không bị dàn trải trên màn hình lớn */
    .block-container {max-width: 1080px; padding-top: 2.2rem; padding-bottom: 2.5rem;}
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* Tiêu đề */
    h1 {font-size: 1.7rem !important; font-weight: 700 !important; letter-spacing: -0.01em;
        padding: 0 0 .25rem 0 !important;}
    h3 {font-size: 1.15rem !important; font-weight: 600 !important; padding-top: .5rem !important;}

    /* Tab */
    button[data-baseweb="tab"] {font-size: .95rem; font-weight: 500; padding: .5rem 1rem;}

    /* Thẻ số liệu */
    [data-testid="stMetric"] {
        background: rgba(128,128,128,.06); border: 1px solid rgba(128,128,128,.22);
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] p {font-size: .8rem; opacity: .75;}
    [data-testid="stMetricValue"] {font-size: 1.35rem; font-weight: 600;}

    /* Nút bấm */
    .stButton > button, .stDownloadButton > button {border-radius: 6px; font-weight: 500; padding: .45rem 1.4rem;}
    .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
        background: #1d3557; border-color: #1d3557; color: #fff;
    }
    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #274472; border-color: #274472; color: #fff;
    }

    /* Thông báo, expander, bảng */
    [data-testid="stAlert"] {border-radius: 8px; padding: .7rem 1rem;}
    [data-testid="stExpander"] {border-radius: 8px; border: 1px solid rgba(128,128,128,.22);}
    [data-testid="stDataFrame"] {border: 1px solid rgba(128,128,128,.22); border-radius: 8px;}

    /* Khối kết luận xu hướng */
    .trend-box {padding: 12px 16px; border-radius: 8px; border-left: 5px solid; margin: 14px 0 10px;
                font-size: 1.05rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


class DataError(Exception):
    """Lỗi có thông báo thân thiện, hiển thị thẳng cho người dùng."""


# ======================= LOAD MODEL / SCALER =======================
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_resources():
    from tensorflow.keras.models import load_model   # import tại đây để báo lỗi thân thiện nếu thiếu thư viện

    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    _, timestep, n_features = model.input_shape       # ví dụ (None, 60, 1)
    if int(n_features) != 1 or getattr(scaler, "n_features_in_", 1) != 1:
        raise DataError("Ứng dụng chỉ hỗ trợ mô hình dùng 1 đặc trưng (giá Close).")
    return {"model": model, "scaler": scaler, "timestep": int(timestep)}


# ======================= KIỂM TRA DỮ LIỆU (DÙNG CHUNG) =======================
def validate_close_data(df: pd.DataFrame, timestep: int, online: bool = False) -> pd.DataFrame:
    """Kiểm tra và chuẩn hóa dữ liệu -> DataFrame[Date, Close] sắp xếp tăng dần theo Date."""
    who = "Dữ liệu từ Yahoo Finance" if online else "File CSV"
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "date" not in cols:
        raise DataError(f"{who} thiếu cột Date.")
    if "close" not in cols:
        raise DataError(f"{who} thiếu cột Close.")

    n_raw = len(df)
    if n_raw < timestep:
        if online:
            raise DataError(f"Chỉ lấy được {n_raw} dòng dữ liệu, nhưng model cần tối thiểu {timestep} dòng.")
        raise DataError(f"File không đủ dữ liệu. File của bạn có {n_raw} dòng, "
                        f"nhưng model cần tối thiểu {timestep} dòng dữ liệu.")

    data = pd.DataFrame({"Date": df[cols["date"]], "Close": df[cols["close"]]})
    raw_close = data["Close"]
    empty = raw_close.isna()
    if raw_close.dtype == object:
        converted = pd.to_numeric(raw_close.str.replace(",", "", regex=False), errors="coerce")
    else:
        converted = pd.to_numeric(raw_close, errors="coerce")

    bad = converted.isna() & ~empty
    if bad.any():
        raise DataError(f"Cột Close có {int(bad.sum())} giá trị không phải số. "
                        "Cột Close chỉ được chứa số (ví dụ 1820.5).")
    data["Close"] = converted

    if empty.any():
        if online:      # dữ liệu online đôi khi có ngày thiếu giá -> bỏ các ngày đó
            data = data[~empty]
        else:
            raise DataError(f"Cột Close đang có {int(empty.sum())} ô bị trống. Vui lòng điền đầy đủ giá trị.")

    dates = pd.to_datetime(data["Date"], errors="coerce")
    if dates.isna().any():
        raise DataError("Cột Date có giá trị không đọc được. Hãy dùng định dạng ngày như 2026-09-01.")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    data["Date"] = dates

    data = data.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if len(data) < timestep:
        raise DataError(f"Dữ liệu hợp lệ chỉ còn {len(data)} dòng, model cần tối thiểu {timestep} dòng.")
    if (data["Close"] <= 0).any():
        raise DataError("Cột Close có giá trị nhỏ hơn hoặc bằng 0, dữ liệu không hợp lệ.")
    return data


# ======================= DỰ ĐOÁN (DÙNG CHUNG) =======================
def predict_price(close: np.ndarray, res: dict):
    """Trả về (giá dự đoán thực tế, cờ 'giá nằm ngoài khoảng model đã học')."""
    timestep, scaler, model = res["timestep"], res["scaler"], res["model"]
    window = close[-timestep:].reshape(-1, 1)             # đúng TIMESTEP giá gần nhất
    try:
        scaled = scaler.transform(window)                 # chỉ transform, KHÔNG fit
        out_of_range = bool(scaled.min() < 0 or scaled.max() > 1)
        x = scaled.reshape(1, timestep, 1)                # (1, TIMESTEP, 1)
        pred_scaled = np.asarray(model.predict(x, verbose=0))
        if pred_scaled.size != 1:
            raise DataError("Dữ liệu không phù hợp với mô hình.")
        predicted = float(scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])
    except ValueError:
        log.exception("Lỗi shape khi dự đoán")
        raise DataError("Dữ liệu không phù hợp với mô hình.")
    return predicted, out_of_range


def build_result(data: pd.DataFrame, res: dict) -> dict:
    current = float(data["Close"].iloc[-1])
    predicted, out_of_range = predict_price(data["Close"].to_numpy(dtype=float), res)
    difference = predicted - current
    percent = difference / current * 100
    r_pred, r_cur = round(predicted, 2), round(current, 2)    # so sánh theo số đang hiển thị
    trend = "up" if r_pred > r_cur else "down" if r_pred < r_cur else "flat"
    last_date = data["Date"].iloc[-1]
    return {
        "data": data, "current": current, "predicted": predicted, "difference": difference,
        "percent": percent, "trend": trend, "out_of_range": out_of_range,
        "last_date": last_date, "next_date": pd.bdate_range(last_date + pd.Timedelta(days=1), periods=1)[0],
    }


# ======================= HIỂN THỊ KẾT QUẢ (DÙNG CHUNG) =======================
TREND_STYLE = {"up": "#1b9e5a", "down": "#d62839", "flat": "#6c757d"}


def render_result(r: dict, texts: dict, chart_days: int, current_label: str):
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric(current_label, f"{r['current']:,.2f}")
    c2.metric("Giá dự đoán", f"{r['predicted']:,.2f}")
    c3.metric("Chênh lệch", f"{r['difference']:+,.2f}")
    c4.metric("Thay đổi", f"{r['percent']:+.2f}%")

    color = TREND_STYLE[r["trend"]]
    st.markdown(
        f"<div class='trend-box' style='background:{color}14;border-color:{color};color:{color}'>"
        f"{texts[r['trend']]}</div>",
        unsafe_allow_html=True)

    if r["out_of_range"]:
        st.warning("Giá vàng gần đây nằm ngoài khoảng giá mà mô hình đã học khi huấn luyện, "
                   "nên kết quả dự đoán có thể kém tin cậy.")

    data = r["data"].tail(chart_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Giá Close thực tế",
                             line=dict(color="#c9a227", width=2)))
    fig.add_trace(go.Scatter(x=[r["last_date"], r["next_date"]], y=[r["current"], r["predicted"]], mode="lines",
                             line=dict(color=color, dash="dash", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["last_date"]], y=[r["current"]], mode="markers+text", name="Giá mới nhất",
                             marker=dict(size=9, color="#1d3557"), text=[f"{r['current']:,.2f}"],
                             textposition="top left"))
    fig.add_trace(go.Scatter(x=[r["next_date"]], y=[r["predicted"]], mode="markers+text", name="Giá dự đoán",
                             marker=dict(size=11, symbol="diamond", color=color), text=[f"{r['predicted']:,.2f}"],
                             textposition="top right"))
    fig.update_layout(
        height=360, hovermode="x unified", template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Giá", gridcolor="rgba(128,128,128,.2)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_error(e: Exception, fallback: str):
    if isinstance(e, DataError):
        st.error(str(e))
    else:
        log.exception("Lỗi không mong đợi")
        st.error(fallback)


# ======================= CSV MẪU =======================
def make_sample_csv(timestep: int) -> bytes:
    n = timestep + 10
    rng = np.random.RandomState(42)
    close = np.round(np.concatenate([[1820.5, 1825.3, 1819.8],
                                     1819.8 + np.cumsum(rng.normal(0, 4, n - 3))]), 2)
    dates = pd.bdate_range("2026-09-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"Date": dates, "Close": close}).to_csv(index=False).encode("utf-8")


# ======================= TAB 1: DỰ ĐOÁN TỪ CSV =======================
def tab_csv(res: dict):
    timestep = res["timestep"]
    st.subheader("Dự đoán từ dữ liệu CSV")

    with st.expander("Hướng dẫn chuẩn bị file CSV"):
        st.markdown(
            f"- File phải có định dạng **.csv**\n"
            f"- Phải có cột **Date** (ngày) và cột **Close** (giá đóng cửa)\n"
            f"- Tối thiểu **{timestep} dòng** dữ liệu\n"
            f"- Cột Close chỉ chứa **số**, không để trống\n"
            f"- Nên sắp xếp theo thứ tự thời gian (app sẽ tự sắp xếp nếu chưa đúng)")
        st.markdown("**Ví dụ:**")
        st.code("Date,Close\n2026-09-01,1820.5\n2026-09-02,1825.3\n2026-09-03,1819.8", language="text")
        st.download_button("Tải file CSV mẫu", make_sample_csv(timestep), "gold_sample.csv", "text/csv")
        st.caption("File mẫu chỉ là dữ liệu minh họa định dạng. Hãy thay bằng giá vàng thật của bạn trước khi dự đoán.")

    choice = st.radio("Nguồn dữ liệu", ["CSV có sẵn", "Tải file CSV của bạn"], horizontal=True)

    if choice == "CSV có sẵn":
        if not CSV_PATH.exists():
            st.error("Không tìm thấy file goldstock.csv trong thư mục dự án.")
            return
        source, name = CSV_PATH, CSV_PATH.name
    else:
        source = st.file_uploader("Chọn file CSV", type=["csv"])
        if source is None:
            st.info("Vui lòng chọn một file CSV để bắt đầu.")
            return
        name = source.name

    try:
        raw = pd.read_csv(source)
    except Exception:
        log.exception("Không đọc được CSV")
        st.error("Không thể đọc file. Hãy kiểm tra file có đúng định dạng CSV hay không.")
        return

    try:
        data = validate_close_data(raw, timestep)
    except DataError as e:
        st.error(str(e))
        return

    st.success("File CSV hợp lệ")
    i1, i2, i3 = st.columns([2, 1, 2])
    i1.markdown(f"**Tên file**  \n{name}")
    i2.markdown(f"**Số dòng**  \n{len(raw):,}")
    i3.markdown(f"**Khoảng thời gian**  \n{data['Date'].iloc[0]:%d/%m/%Y} → {data['Date'].iloc[-1]:%d/%m/%Y}")
    preview = data.tail(10).copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d")
    st.caption("10 dòng dữ liệu gần nhất")
    st.dataframe(preview, hide_index=True, use_container_width=True, height=250)

    signature = f"{choice}|{name}|{len(raw)}|{data['Close'].iloc[-1]}"
    if st.button("Dự đoán", type="primary"):
        try:
            with st.spinner("Đang dự đoán..."):
                st.session_state["csv_result"] = (signature, build_result(data, res))
        except Exception as e:
            st.session_state.pop("csv_result", None)
            show_error(e, "Có lỗi khi dự đoán. Dữ liệu có thể không phù hợp với mô hình.")

    saved = st.session_state.get("csv_result")
    if saved and saved[0] == signature:
        render_result(saved[1],
                      {"up": "Dự đoán giá vàng tăng", "down": "Dự đoán giá vàng giảm",
                       "flat": "Dự đoán giá vàng đi ngang"},
                      chart_days=120, current_label="Giá vàng gần nhất")


# ======================= TAB 2: DỰ ĐOÁN GIÁ VÀNG HIỆN TẠI =======================
def fetch_online(timestep: int) -> pd.DataFrame:
    """Lấy GC=F, chu kỳ 1 ngày, chỉ giữ Date và Close, rồi validate như Tab 1."""
    try:
        import yfinance as yf
    except ImportError:
        raise DataError("Thiếu thư viện yfinance. Hãy chạy: pip install yfinance")
    try:
        raw = yf.download(TICKER, period="1y", interval="1d", progress=False, auto_adjust=False)
    except Exception:
        log.exception("Lỗi yfinance")
        raise DataError("Không thể lấy dữ liệu giá vàng mới nhất. Vui lòng kiểm tra kết nối Internet và thử lại.")
    if raw is None or raw.empty:
        raise DataError("Không nhận được dữ liệu giá vàng. Vui lòng kiểm tra kết nối Internet và thử lại sau.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    return validate_close_data(raw[[c for c in ("Date", "Close") if c in raw.columns]], timestep, online=True)


def tab_online(res: dict):
    timestep = res["timestep"]
    st.subheader("Dự đoán giá vàng hiện tại")
    st.write("Lấy giá vàng mới nhất từ Internet và dự đoán giá phiên giao dịch kế tiếp.")

    if st.button("Cập nhật dữ liệu mới nhất", type="primary"):
        try:
            with st.spinner("Đang lấy dữ liệu giá vàng mới nhất..."):
                data = fetch_online(timestep)
            with st.spinner("Đang dự đoán..."):
                result = build_result(data, res)
            st.session_state["online_result"] = (datetime.now().strftime("%H:%M:%S %d/%m/%Y"), result)
        except Exception as e:
            st.session_state.pop("online_result", None)
            show_error(e, "Có lỗi khi cập nhật dữ liệu. Vui lòng thử lại.")

    saved = st.session_state.get("online_result")
    if saved is None:
        st.info("Nhấn nút **Cập nhật dữ liệu mới nhất** để lấy giá vàng và dự đoán.")
        return
    updated_at, r = saved
    st.caption(f"Cập nhật lúc {updated_at} · Giá Close mới nhất ngày {r['last_date']:%d/%m/%Y} · "
               f"Nguồn: Yahoo Finance ({TICKER})")
    render_result(r, {"up": "Dự đoán tăng", "down": "Dự đoán giảm", "flat": "Dự đoán đi ngang"},
                  chart_days=timestep, current_label="Giá vàng hiện tại")


# ======================= GIAO DIỆN CHÍNH =======================
st.title("Dự đoán giá vàng bằng LSTM")
st.write("Ứng dụng sử dụng mô hình LSTM để dự đoán giá vàng dựa trên dữ liệu lịch sử.")
st.caption("Chỉ phục vụ học tập, không phải khuyến nghị đầu tư.")

missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
if missing:
    st.error(f"Không tìm thấy file: {', '.join(missing)}. Hãy đặt file cùng thư mục với app.py.")
    st.stop()

try:
    resources = load_resources()
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception:
    log.exception("Không load được model/scaler")
    st.error("Không thể tải mô hình. File có thể bị lỗi hoặc không tương thích với phiên bản thư viện đang cài.")
    st.stop()

tab1, tab2 = st.tabs(["Dự đoán từ CSV", "Dự đoán giá vàng hiện tại"])
with tab1:
    tab_csv(resources)
with tab2:
    tab_online(resources)