#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple script to start a VLLM server for local model evaluation.

Usage:
    python games/evaluation/start_vllm.py --model-path /path/to/model

The server will run until interrupted (Ctrl+C). Make sure your config file
uses the correct URL (e.g., http://localhost:8000/v1) and model_name.
"""
import argparse
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Start a VLLM server for local model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start VLLM server with default settings
  python games/evaluation/start_vllm.py --model-path /path/to/model
  
  # Start with custom port and model name
  python games/evaluation/start_vllm.py \\
      --model-path /path/to/model \\
      --port 8000 \\
      --model-name local_model
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="local_model",
        help="Model name to serve (default: local_model)",
    )
    
    # Allow additional vllm arguments
    args, unknown_args = parser.parse_known_args()
    
    # Check model path
    if not Path(args.model_path).exists():
        print(f"Error: Model path not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Build vllm serve command
    cmd = [
        "vllm",
        "serve",
        args.model_path,
        "--served-model-name", args.model_name,
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    # Add any additional arguments passed through
    cmd.extend(unknown_args)
    
    print(f"Starting VLLM server: {' '.join(cmd)}")
    print(f"\nServer will be available at: http://{args.host}:{args.port}/v1")
    print(f"Model name: {args.model_name}")
    print(f"\nConfigure your config file with:")
    print(f"  default_model:")
    print(f"    url: http://{args.host}:{args.port}/v1")
    print(f"    model_name: {args.model_name}")
    print(f"\nPress Ctrl+C to stop the server\n")
    
    # Start server and wait
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down VLLM server...")
        process.terminate()
        process.wait()
    except FileNotFoundError:
        print("Error: vllm command not found. Please install vllm first: pip install vllm", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
