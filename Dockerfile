FROM continuumio/miniconda3:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG CONDA_ENV=sklearn-porter

ARG PYTHON_VER
ARG CYTHON_VER
ARG NUMPY_VER
ARG SCIPY_VER
ARG SKLEARN_VER

# Basics:
RUN echo "deb http://deb.debian.org/debian/ sid main" >> /etc/apt/sources.list && \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    apt-transport-https apt-utils curl make

# Java:
RUN mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends \
    openjdk-8-jdk=8u232-b09-1

# GCC:
RUN apt-get install -y --no-install-recommends \
    libopenblas-dev liblapack-dev \
    gfortran=4:8.3.0-1 \
    cpp=4:8.3.0-1 \
    g++=4:8.3.0-1 \
    gcc=4:8.3.0-1

# Ruby:
RUN apt-get install -y --no-install-recommends \
    ruby=1:2.5.1

# PHP:
RUN apt-get install -y --no-install-recommends \
    php=2:7.3+69

# Node.js:
RUN apt-get install -y --no-install-recommends \
    nodejs

# Go:
RUN mkdir -p /tmp/go && cd /tmp/go \
    && curl --silent -o go.tar.gz https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz \
    && tar -xf go.tar.gz \
    && mv go /usr/bin
ENV PATH="/usr/bin/go/bin:${PATH}"

RUN mkdir -p /app
WORKDIR /app
COPY . /app

RUN env | grep _VER \
    && conda update -y -n base conda \
    && conda create -y -n ${CONDA_ENV} ${PYTHON_VER:-python=3.5} \
    && conda run -n ${CONDA_ENV} pip install --upgrade pip \
    && conda run -n ${CONDA_ENV} pip install ${CYTHON_VER:-cython} \
    && conda run -n ${CONDA_ENV} pip install ${NUMPY_VER:-numpy} \
    && conda run -n ${CONDA_ENV} pip install ${SCIPY_VER:-scipy} \
    && conda run -n ${CONDA_ENV} pip install ${SKLEARN_VER:-scikit-learn} \
    && conda run -n ${CONDA_ENV} make install.requirements.development \
    && conda env export -n ${CONDA_ENV}