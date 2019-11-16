FROM continuumio/miniconda3:latest

ARG CONDA_ENV=sklearn-porter

ARG PYTHON_VER
ARG CYTHON_VER
ARG NUMPY_VER
ARG SCIPY_VER
ARG SCIKIT_LEARN_VER

COPY . $HOME/app
WORKDIR $HOME/app

RUN mkdir -p /usr/share/man/man1 && \
    echo "deb http://deb.debian.org/debian/ sid main" >> /etc/apt/sources.list && \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    apt-transport-https apt-utils curl make libopenblas-dev liblapack-dev \
    gfortran=4:8.3.0-1 \
    cpp=4:8.3.0-1 \
    g++=4:8.3.0-1 \
    gcc=4:8.3.0-1                                                   `# gcc v8.3.0`      \
    ruby=1:2.5.1                                                    `# ruby v2.5.1`     \
    php=2:7.3+69                                                    `# php v7.3`        \
    openjdk-8-jdk=8u232-b09-1                                       `# java v1.8.x`     \
    && apt-get install -y nodejs                                    `# node v10.x.x`    \
    && wget --quiet https://dl.google.com/go/go1.12.4.linux-amd64.tar.gz \
    && tar -xf go1.12.4.linux-amd64.tar.gz                          `# go v1.12.4`      \
    && mv go /usr/bin \
    && make clean \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/bin/go/bin:${PATH}"

RUN env | grep _VER \
    && conda update -y -n base conda \
    && conda create -y -n ${CONDA_ENV} ${PYTHON_VER:-python=3.5} \
    && conda run -n ${CONDA_ENV} pip install --upgrade pip \
    && conda run -n ${CONDA_ENV} pip install ${CYTHON_VER:-cython} \
    && conda run -n ${CONDA_ENV} pip install ${NUMPY_VER:-numpy} \
    && conda run -n ${CONDA_ENV} pip install ${SCIPY_VER:-scipy} \
    && conda run -n ${CONDA_ENV} pip install ${SCIKIT_LEARN_VER:-scikit-learn} \
    && conda run -n ${CONDA_ENV} make install.requirements.development \
    && conda env export -n ${CONDA_ENV}