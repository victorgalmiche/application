FROM astral/uv:python3.12-bookworm-slim

# Install project dependencies
COPY pyproject.toml .
RUN uv sync

COPY train.py .
COPY src ./src
COPY app ./app
CMD ["bash", "-c", "./app/run.sh"]
