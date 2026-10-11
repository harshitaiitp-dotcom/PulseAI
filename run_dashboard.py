"""
Quick-Start Launcher for ProjectPulse AI Dashboard
Starts the FastAPI server and automatically opens the web dashboard in the default browser.
"""

import os
import sys
import webbrowser
import threading
import time
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\nOpening ProjectPulse AI Dashboard in browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not automatically open browser: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 LAUNCHING PROJECTPULSE AI - ML & DL IN PROJECT ANALYTICS")
    print("=" * 70)
    print("Backend: FastAPI + PyTorch + Scikit-Learn")
    print("Dashboard URL: http://127.0.0.1:8000")
    print("Press Ctrl+C to terminate the server.\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("server.api:app", host="127.0.0.1", port=8000, log_level="info")
