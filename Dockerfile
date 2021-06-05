FROM python:3.8-slim

RUN apt-get -y update && apt-get -y install git wget

RUN mkdir /app

WORKDIR /app

RUN git clone https://github.com/activeloopai/Hub.git && \
    pip install -r Hub/hub/requirements.txt && \
    pip install -r Hub/hub/common.txt && \
    pip install -r Hub/hub/tests.txt && \
    pip install -r Hub/hub/plugins.txt

WORKDIR /app/Hub

RUN pytest .