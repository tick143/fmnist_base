from __future__ import annotations

import argparse

import uvicorn

from .server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the tiny spiking visualisation server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind the server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the HTTP server.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
