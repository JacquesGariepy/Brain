"""
Brain CLI - Serve Command
"""

import sys


def serve_command(args):
    """Execute serve command to start API server"""
    print("=" * 70)
    print("Brain Framework - API Server")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Workers: {args.workers}")
    print(f"  Reload: {args.reload}")

    print("\n" + "-" * 70)

    try:
        # Check if FastAPI is available
        try:
            import uvicorn
            from api.app import app
        except ImportError:
            print("\n✗ Error: FastAPI and uvicorn are required for serving")
            print("   Install with: pip install fastapi uvicorn")
            sys.exit(1)

        print("\nStarting server...")
        print(f"  → API docs: http://{args.host}:{args.port}/docs")
        print(f"  → Health check: http://{args.host}:{args.port}/health")
        print("\nPress Ctrl+C to stop the server\n")

        # Start server
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
        )

    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error starting server: {e}", file=sys.stderr)
        sys.exit(1)
