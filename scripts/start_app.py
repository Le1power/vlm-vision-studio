"""一条命令启动 Streamlit 界面。"""

from __future__ import annotations

import sys
import os
from pathlib import Path


def main() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().with_name("run_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
