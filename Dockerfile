FROM continuumio/miniconda3:4.6.14

ARG DEBIAN_FRONTEND=noninteractive

ENV CONDA_ENV sklearn-porter

ARG PYTHON_VER
ARG CYTHON_VER
ARG NUMPY_VER
ARG SCIPY_VER
ARG SKLEARN_VER

ARG NB_USER=jovyan
ARG NB_UID=1000

ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# ---------------------------------------------

# Basics:
RUN echo "deb http://deb.debian.org/debian/ sid main" >> /etc/apt/sources.list && \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    apt-transport-https apt-utils curl make

# Java:
RUN mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends \
    openjdk-8-jdk

# GCC:
RUN apt-get install -y --no-install-recommends \
    libopenblas-dev liblapack-dev gfortran cpp g++ gcc

# Ruby:
RUN apt-get install -y --no-install-recommends ruby

# PHP:
RUN apt-get install -y --no-install-recommends php

# Node.js:
RUN apt-get install -y --no-install-recommends nodejs

# Go:
RUN mkdir -p /tmp/go && cd /tmp/go \
    && curl --silent -o go.tar.gz https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz \
    && tar -xf go.tar.gz \
    && mv go /usr/bin
ENV PATH="/usr/bin/go/bin:${PATH}"

RUN java -version \
    && gcc --version \
    && ruby --version \
    && php --version \
    && python --version \
    && node --version \
    && go version

# ---------------------------------------------

RUN adduser --disabled-password \
    --gecos "default user" \
    --shell /bin/bash \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}

RUN chown -R ${NB_UID} ${HOME}
RUN chmod 755 ${HOME}/docker-entrypoint.sh

WORKDIR ${HOME}

USER ${NB_USER}

# Install each dependency step by step:
RUN conda create -y -n ${CONDA_ENV} ${PYTHON_VER:-python=3.5}
RUN conda run -n ${CONDA_ENV} python -m pip install --upgrade pip
RUN conda run -n ${CONDA_ENV} python -m pip install ${CYTHON_VER:-cython}
RUN conda run -n ${CONDA_ENV} python -m pip install ${NUMPY_VER:-numpy}
RUN conda run -n ${CONDA_ENV} python -m pip install ${SCIPY_VER:-scipy}
RUN conda run -n ${CONDA_ENV} python -m pip install ${SKLEARN_VER:-scikit-learn}
RUN conda run -n ${CONDA_ENV} python -m pip install --no-cache-dir -e .[development,examples]

# Extend system path for the notebooks:
RUN conda run -n ${CONDA_ENV} ipython profile create \
    && echo -e "\nc.InteractiveShellApp.exec_lines = [\x27import sys; sys.path.append(\x22${HOME}\x22)\x27]" >> $(conda run -n ${CONDA_ENV} ipython locate)/profile_default/ipython_config.py

ENTRYPOINT ["./docker-entrypoint.sh"]