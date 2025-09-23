import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# Backend FastAPI endpoint
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Model location check
FASTAPI_WINE_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'model' / 'wine_model.pkl'

# streamlit logger
LOGGER = get_logger(__name__)

def run():
    # Page settings
    st.set_page_config(
        page_title="Wine Classification Demo",
        page_icon="🍷",
    )

    # Sidebar
    with st.sidebar:
        # Backend status check
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if backend_request.status_code == 200:
                st.success("Backend online ✅")
            else:
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            st.error("Backend offline 😱")

        st.info("Upload JSON input for wine prediction")

        # File uploader
        test_input_file = st.file_uploader('Upload test prediction file', type=['json'])

        if test_input_file:
            st.write('Preview file')
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
        else:
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False

        predict_button = st.button('Predict')

    # Main body
    st.write("# 🍷 Wine Class Prediction")

    if predict_button:
        if "IS_JSON_FILE_AVAILABLE" in st.session_state and st.session_state["IS_JSON_FILE_AVAILABLE"]:
            if FASTAPI_WINE_MODEL_LOCATION.is_file():
                client_input = json.dumps(test_input_data)
                try:
                    result_container = st.empty()
                    with st.spinner('Predicting...'):
                        predict_wine_response = requests.post(f'{FASTAPI_BACKEND_ENDPOINT}/predict', client_input)

                    if predict_wine_response.status_code == 200:
                        wine_content = json.loads(predict_wine_response.content)
                        pred_class = wine_content["response"]
                        result_container.success(f"Predicted Wine Class: {pred_class}")
                    else:
                        st.toast(f':red[Status from server: {predict_wine_response.status_code}. Refresh page and check backend status]', icon="🔴")
                except Exception as e:
                    st.toast(':red[Problem with backend. Refresh page and check backend status]', icon="🔴")
                    LOGGER.error(e)
            else:
                LOGGER.warning('wine_model.pkl not found in FastAPI Lab. Make sure to run train.py')
                st.toast(':red[Model wine_model.pkl not found. Please run train.py in src-wine]', icon="🔥")
        else:
            LOGGER.error('Provide a valid JSON file with all wine features')
            st.toast(':red[Please upload a JSON test file with all wine dataset features]', icon="📂")

if __name__ == "__main__":
    run()
