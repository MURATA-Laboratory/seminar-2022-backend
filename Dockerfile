FROM python:3.10.5

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONPATH=/src

EXPOSE 8000

WORKDIR /src

RUN apt update \
    && apt install -y --no-install-recommends less

RUN pip install poetry \
    && poetry config virtualenvs.create false

COPY ./app/healthcheck.py ./pyproject.toml ./poetry.lock ./
RUN poetry install --no-dev --no-root

CMD ["poetry","run","uvicorn","healthcheck:app","--reload","--host","0.0.0.0","--port","8000"]