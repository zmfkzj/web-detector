FROM python:3.12-slim


COPY ./poetry.lock ./pyproject.toml /app/

RUN pip install poetry && poetry install

WORKDIR /app

COPY . .

WORKDIR /app/shoes_detector

CMD [ "poetry", "run", "uvicorn", "app:app", "--host=0.0.0.0", "--ssl-keyfile=../key.pem", "--ssl-certfile=../cert.pem" ]
