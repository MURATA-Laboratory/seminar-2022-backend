FROM python:3.8-slim as builder

RUN mkdir /resource

WORKDIR /resource

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt > requirements.txt


FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /resource

COPY --from=builder /resource/requirements.txt .

RUN pip install -r requirements.txt

COPY ./healthcheck.py /resource/healthcheck.py

EXPOSE 8000

CMD [ "uvicorn", "healthcheck:resource", "--host", "0.0.0.0" ]