"""CLI entry point for Tinker Chef.

Usage::

    tinker-chef serve /path/to/log_dir
    tinker-chef serve /path/to/log_dir --port 8150 --host 0.0.0.0
"""

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tinker-chef",
        description="Tinker Chef — training visualization dashboard for tinker-cookbook",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument(
        "log_dir",
        help="Path to a training run directory or a parent directory containing multiple runs",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8150,
        help="Port to bind to (default: 8150)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _run_serve(args)


def _run_serve(args: argparse.Namespace) -> None:
    """Start the Tinker Chef server."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed. Install chef dependencies with:\n"
            "  pip install tinker_cookbook[chef]",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from tinker_cookbook.chef.app import create_app

    app = create_app(args.log_dir)

    print(f"\n  Tinker Chef starting on http://{args.host}:{args.port}")
    print(f"  Serving runs from: {args.log_dir}")
    print(f"  API docs at: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
