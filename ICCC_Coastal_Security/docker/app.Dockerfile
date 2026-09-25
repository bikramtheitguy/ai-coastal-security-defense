# Multi-stage build: (1) static Next.js export, (2) Python runtime serving API + frontend.
# Base images are build args so a registry mirror can be used (e.g. --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.11-slim).
ARG NODE_IMAGE=node:22-alpine
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${NODE_IMAGE} AS frontend
COPY docker/certs/ /tmp/certs/
RUN cat /tmp/certs/*.crt > /tmp/extra-ca.pem 2>/dev/null || true
ENV NODE_EXTRA_CA_CERTS=/tmp/extra-ca.pem
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npx next build

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN (ls /usr/local/share/ca-certificates/extra/*.crt >/dev/null 2>&1 && update-ca-certificates) || true
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt && useradd --create-home --uid 10001 iccc
COPY backend/ backend/
COPY seed/ seed/
COPY assets/ assets/
COPY --from=frontend /src/frontend/out frontend/out
RUN mkdir -p /app/data && chown -R iccc /app/data
USER iccc
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/public/info')"
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
