import streamlit as st
import requests
import time
import plotly.graph_objects as go

# ====================== 1. Page config ======================
st.set_page_config(
    page_title="Cognitive Shorts Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== 2. Session state (persists across reruns) ======================
# Record app start time (initialize only once)
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# Backend API base URL (globally available)
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

# Current page (for button navigation)
if "current_page" not in st.session_state:
    st.session_state.current_page = "Single Prediction"

# ====================== 3. Generic API request helper ======================
def call_api(endpoint, method="GET", payload=None):
    """Call the backend API.

    endpoint: path (e.g. "predict" / "health")
    method:   "GET" or "POST"
    payload:  data to send for POST
    Returns (data, error): one is None.
    """
    url = f"{st.session_state.api_url}/{endpoint}"

    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=10)
        else:
            response = requests.get(url, timeout=5)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "API server is offline. Please start src/api.py"

    except Exception as e:
        return None, f"Error: {str(e)}"
    
# ====================== 4. Sidebar ======================
def render_sidebar():
    with st.sidebar:
        st.header("📋 Navigation")
        st.divider()

        # Navigation buttons (highlight the active page)
        pages = ["Single Prediction", "Batch Prediction", "Analytics Dashboard", "Model Info"]
        for page in pages:
            is_active = st.session_state.current_page == page
            if st.button(
                page,
                use_container_width=True,
                type='primary' if is_active else 'secondary',
            ):
                st.session_state.current_page = page
        st.divider()

        # API status
        st.subheader("API Status")
        data, err = call_api('health')
        if data and data.get('status') == 'healthy':
            st.success("✅ Healthy")
        else:
            st.error("❌ Offline")
        
        # Show uptime
        st.subheader("Uptime")
        uptime = int(time.time() - st.session_state.start_time)
        st.markdown(f"# {uptime}s")

# ====================== 5. Single Prediction page ======================
def page_single_prediction():
    st.header("🎯 Single Prediction")
    st.caption("Enter user and video information to predict whether a like will be given")
    st.divider()

    col_input, col_result = st.columns([1.5, 1])

    # ---------- Left: input parameters ----------
    with col_input:
        st.subheader("Input Parameters")

        user_id = st.text_input("User ID", value="user_000005")
        video_id = st.text_input("Video ID", value="video_0000001")
        watch_time = st.slider("Watch Time (seconds)", 0.0, 300.0, 7.06)
        hour_of_day = st.number_input("Hour of Day", min_value=0, max_value=23, value=14)

        with st.expander("Advanced Options"):
            st.selectbox("Model Version", ["v1.0.0 (Production)", "v1.1.0 (Staging)"])

    # ---------- Right: prediction result ----------
    with col_result:
        st.subheader("Prediction Result")

        predict_btn = st.button("🚀 Start Prediction", type="primary", use_container_width=True)

        if predict_btn:
            payload = {
                "user_id": user_id,
                "video_id": video_id,
                "watch_time_seconds": watch_time,
                "hour_of_day": hour_of_day,
            }

            with st.spinner("Analyzing..."):
                start_ts = time.time()
                result, err = call_api("predict", method="POST", payload=payload)
                latency = (time.time() - start_ts) * 1000

            if result:
                prob = result["probability"]
                pred = result["prediction"]
                conf = result["confidence"]

                if pred == 1:
                    st.success(f"✅ User 【{user_id}】 will like this video")
                else:
                    st.error(f"❌ User 【{user_id}】 will not like this video")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    title={"text": "Like Probability"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "royalblue"},
                        "steps": [
                            {"range": [0, 0.5], "color": "tomato"},
                            {"range": [0.5, 0.8], "color": "lightgreen"},
                            {"range": [0.8, 1], "color": "green"},
                        ],
                    },
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Three metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Probability", f"{prob:.2%}")
                m2.metric("Confidence", conf)
                m3.metric("Latency", f"{latency:.0f}ms")
            else:
                st.error(err)
# ====================== 6. Placeholder pages ======================
def page_batch_prediction():
    st.header("🗃️ Batch Prediction")
    st.info("Coming soon.")


def page_analytics_dashboard():
    st.header("📊 Analytics Dashboard")
    st.info("Coming soon.")


def page_model_info():
    st.header("ℹ️ Model Info")
    st.info("Coming soon.")


# ====================== 7. Main (routing) ======================
def main():
    render_sidebar()
    st.divider()

    page = st.session_state.current_page
    if page == "Single Prediction":
        page_single_prediction()
    elif page == "Batch Prediction":
        page_batch_prediction()
    elif page == "Analytics Dashboard":
        page_analytics_dashboard()
    elif page == "Model Info":
        page_model_info()


if __name__ == "__main__":
    main()