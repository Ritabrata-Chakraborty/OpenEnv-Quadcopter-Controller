# Quick Reference: Run Inference in Docker

> Run all commands from inside the `quadnav/` directory.

## Full Workflow

```bash
# 1. Clean Images & Containers
docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null
docker rmi -f $(docker images -q) 2>/dev/null

# 2. Build Image (from project root, one level up)
cd ..
docker build -t quadnav:latest quadnav/
cd quadnav

# 3. Run Container (env vars passed from .env file)
docker run -d \
  --name quadnav-env \
  -p 8000:8000 \
  --env-file .env \
  -e QUADNAV_ENV_URL=http://localhost:8000 \
  quadnav:latest

# Wait for server to be ready
sleep 10

# 4. Run Inference (no .env copy needed)
docker exec -w /app/env quadnav-env bash -c "
uv run python3 inference.py
"

# 5. Cleanup
docker stop quadnav-env && docker rm quadnav-env
```