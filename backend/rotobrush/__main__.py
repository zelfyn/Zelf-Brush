"""python -m rotobrush [--port 8765]"""
import argparse

import uvicorn

from .server import app


def main() -> None:
    ap = argparse.ArgumentParser(description="Zelf-Brush processing backend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
