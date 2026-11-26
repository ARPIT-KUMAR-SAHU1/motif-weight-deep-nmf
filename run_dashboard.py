"""Launch the local SG-DMNMF results dashboard.

    python run_dashboard.py

The dashboard opens at http://127.0.0.1:8000 (or the next free port).
If a browser does not open automatically, copy that URL into your browser.
Set PORT to choose a port and NO_BROWSER=1 to skip auto-opening.
"""
import os
import socket

from ui.dashboard import app

ROOT = os.path.dirname(os.path.abspath(__file__))


def _free_port(preferred):
    """Return the preferred port if free, otherwise the next free port."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port found near {preferred}")


def main():
    port = _free_port(int(os.environ.get("PORT", "8000")))
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 64)
    print("  SG-DMNMF DASHBOARD IS RUNNING")
    print(f"  Open this address in your browser:  {url}")
    print("  (If no browser opens automatically, paste the address yourself.)")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 64 + "\n", flush=True)

    if os.environ.get("NO_BROWSER", "0") != "1":
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False,
            threaded=True)


if __name__ == "__main__":
    main()
