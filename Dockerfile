FROM python:3.10

ENV KAFKA_URI=localhost:9094

WORKDIR /home/app

COPY . /home/app/

RUN pip install -e .

CMD [ "python", "./src/index.py" ]