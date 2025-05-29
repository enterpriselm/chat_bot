FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG GENAITOR_REPO_URL
ARG GOOGLE_API_KEY_ARG

RUN git clone ${GENAITOR_REPO_URL} genaitor

WORKDIR /app/genaitor

RUN pip install -e .

COPY app.py .

RUN echo "API_KEY=${GOOGLE_API_KEY_ARG}" > .env

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
