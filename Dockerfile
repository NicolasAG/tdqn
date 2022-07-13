FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y --fix-missing && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential libffi-dev python3-dev curl git redis-server && \
    apt-get autoremove -y --purge && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U redis sentencepiece
RUN pip install tensorflow==1.15
RUN pip install spacy==2.3
RUN python -m spacy download en
RUN pip install jericho==2.1.0

