FROM python:3.8-slim

RUN apt-get -y update && apt-get -y install git wget

RUN mkdir /app

WORKDIR /app

RUN git clone https://github.com/activeloopai/Hub.git && \
    cd Hub && \
    git checkout clean_for_release

RUN pip install -r requirements/all-requirements.txt && \
    pip install -r requirements/common.txt && \
    pip install -r requirements/tests.txt && \
    pip install -r requirements/plugins.txt

WORKDIR /app/Hub

RUN pytest .