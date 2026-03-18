"""Deployment compatibility entrypoint.

Supports:
- streamlit run main.py
- ASGI loaders expecting main:api (or main:app)
"""

from ml.main import app as app
from ml.main import run_streamlit_ui

# Some platforms are configured to look for `main:api`.
api = app


if __name__ == "__main__":
    run_streamlit_ui()
