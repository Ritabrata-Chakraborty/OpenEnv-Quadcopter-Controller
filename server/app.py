"""
FastAPI application for the Quadnav Env Environment.

This module creates an HTTP server that exposes the QuadnavEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m quadnav.server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import SchemaResponse
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import QuadnavAction, QuadnavObservation, QuadnavState
    from .environment import QuadnavEnvironment
except ModuleNotFoundError:
    from quadnav.models import QuadnavAction, QuadnavObservation, QuadnavState
    from quadnav.server.environment import QuadnavEnvironment


# Create the app with web interface and README integration
app = create_app(
    QuadnavEnvironment,
    QuadnavAction,
    QuadnavObservation,
    env_name="quadnav_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# Patch endpoint to include QuadnavState in schema
# We replace the existing /schema endpoint by overwriting its function
from starlette.routing import Route
from typing import List

# Find and remove the old /schema GET endpoint
new_routes: List = []
for route in app.routes:
    if isinstance(route, Route) and route.path == "/schema" and "GET" in [m.upper() for m in (route.methods or [])]:
        continue
    new_routes.append(route)
app.router.routes = new_routes

# Now add our custom /schema endpoint
@app.get("/schema", tags=["Schema"], response_model=SchemaResponse, include_in_schema=True)
async def get_schemas_with_state() -> SchemaResponse:
    """Get JSON schemas for action, observation, and state with correct QuadnavState."""
    return SchemaResponse(
        action=QuadnavAction.model_json_schema(),
        observation=QuadnavObservation.model_json_schema(),
        state=QuadnavState.model_json_schema(),
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m quadnav.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn quadnav.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
