"""Compatibility shim.

Canonical implementation now lives in root main.py.
This file remains only to keep legacy commands working.
"""

from main import api, app, run_cli, run_streamlit_ui, running_in_streamlit


if __name__ == "__main__":
    if running_in_streamlit():
        run_streamlit_ui()
    else:
        run_cli()
