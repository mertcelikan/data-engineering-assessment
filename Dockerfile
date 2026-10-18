FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" /opt/java \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java \
    PYSPARK_PYTHON=python \
    SPARK_LOCAL_IP=127.0.0.1 \
    PYTHONPATH=/app

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

WORKDIR /app
RUN chown app:app /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app config ./config
COPY --chown=app:app data ./data
COPY --chown=app:app src ./src
COPY --chown=app:app tests ./tests
RUN mkdir -p /app/output && chown app:app /app/output

USER app

CMD ["python", "-m", "src.main"]
