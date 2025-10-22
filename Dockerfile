FROM python:3.12-slim as builder

ARG ENV=prod
ARG TELEGRAM_TOKEN=""
ARG CLIENT=gradio
ENV APP_ENV=$ENV
ENV TELEGRAM_TOKEN=$TELEGRAM_TOKEN
ENV CLIENT=$CLIENT
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-dev libhunspell-dev hunspell-en-us g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip

WORKDIR /nlp_suicide_watch

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true && \
    find /opt/venv -name "tests" -type d -exec rm -rf {} + || true && \
    find /opt/venv -name "test" -type d -exec rm -rf {} + || true && \
    find /opt/venv -name "*.pyx" -delete || true && \
    find /opt/venv -name "*.pxd" -delete || true && \
    find /opt/venv -name "*.c" -delete || true && \
    find /opt/venv -name "*.md" -delete || true && \
    find /opt/venv -name "*.rst" -delete || true && \
    find /opt/venv -name "docs" -type d -exec rm -rf {} + || true && \
    find /opt/venv -name "examples" -type d -exec rm -rf {} + || true

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y libhunspell-dev hunspell-en-us libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /nlp_suicide_watch

COPY --from=builder /opt/venv /opt/venv

RUN python -m spacy download en_core_web_sm

COPY entrypoint.sh /entrypoint.sh
COPY . .

RUN chmod +x /entrypoint.sh

EXPOSE 7860
CMD ["/entrypoint.sh"]
