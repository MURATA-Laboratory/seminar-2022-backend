FROM python:3.8-alpine

WORKDIR /resource

COPY requirements.txt .

RUN apk add --no-cache build-base \
 && pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt \
 && apk del build-base

COPY healthcheck.py .

CMD ["uvicorn","healthcheck:resource","--reload","--host","0.0.0.0","--port","8000"]