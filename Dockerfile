FROM python:3.12-slim

RUN pip install poetry && poetry install

COPY . /app/

WORKDIR /app/shoes_detector

CMD [ "poetry", "run", "fastapi", "run" ]