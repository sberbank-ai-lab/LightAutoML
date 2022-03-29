FROM ubuntu:latest

RUN apt-get update

RUN apt install -y python3-dev python3-pip curl git

RUN git clone https://github.com/sberbank-ai-lab/LightAutoML.git
WORKDIR /LightAutoML/

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH="${PATH}:/root/.poetry/bin"

RUN poetry -V
RUN poetry config virtualenvs.create false --local
RUN poetry install
