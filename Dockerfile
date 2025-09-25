FROM python:3.12-slim


COPY . /app/

WORKDIR /app

RUN pip install poetry && poetry install

WORKDIR /app/shoes_detector

CMD [ "poetry", "run", "uvicorn", "app:app", "--host=0.0.0.0", "--ssl-keyfile=../key.pem", "--ssl-certfile=../cert.pem" ]
