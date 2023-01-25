# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /app

RUN apt-get update && apt-get install -y \
    graphviz

#ADD ./Requirements.txt /Requirements.txt
#ADD ./setup.py /setup.py
#RUN python3 /setup.py
# COMMENT THESE TO MOUNT APP FOR DEV MODE
COPY . /app
# RUN pip install "ray[rllib]" 
# RUN pip install "ray[tune]" 

RUN pip install -U "ray[default]"
RUN pip install ray[rllib]
RUN pip install ray[tune]
RUN pip install ray[air]
RUN pip install ray[serve]
RUN pip install gputil
RUN pip install graphviz


# COMMENT THESE TO MOUNT APP FOR DEV MODE
RUN pip install -e /app

#RUN pip install stable-baselines3[extra]

ENV PYTHONPATH="/app"



# To build the docker image:
#   docker build -t cage:1.0 .
#
# To run in terminal 
#   docker run -it --shm-size=32.00gb --rm cage:1.0 bash
# To run in jupyter 
# docker run -it -v /home/adamprice/Cage-Challenge/CybORG/CybORG:/tf/Cage -p 8888:8888 cage:1.0
# docker run -it -v /home/ubuntu/u75a/u75a-Data-Efficient-Decisions/CybORG/CybORG:/tf/Cage -p 8888:8888 cage:1.0
# 

