# run_streamlit.ps1 - helper to run the app locally
.\.venv\Scripts\activate
pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
