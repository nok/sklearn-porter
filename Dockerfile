FROM continuumio/miniconda3:latest

ARG CONDA_ENV=sklearn-porter

COPY . $HOME/app
WORKDIR $HOME/app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    make=4.1-9.1 \
    g++=4:6.3.0-4 \
    gcc=4:6.3.0-4                                                   `# gcc v6.3.0`      \
    ruby=1:2.3.3                                                    `# ruby v2.3.3`     \
    php7.0=7.0.33-0+deb9u3                                          `# php v7.0.33`     \
    openjdk-8-jdk=8u212-b01-1~deb9u1                                `# java v1.8.0`     \
    && curl -sL https://deb.nodesource.com/setup_10.x | bash \
    && apt-get install -y nodejs=10.15.3-1nodesource1               `# node v10.15.3`   \
    && npm install --global xmlhttprequest \
    && wget https://dl.google.com/go/go1.12.4.linux-amd64.tar.gz \
    && tar -xvf go1.12.4.linux-amd64.tar.gz                         `# go v1.12.4`      \
    && mv go /usr/bin \
    && wget http://central.maven.org/maven2/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar \
    && mv gson-2.8.5.jar gson.jar \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN make clean

ENV PATH="/usr/bin/go/bin:${PATH}"

RUN conda update -y -n base conda \
    && conda create -y -n ${CONDA_ENV} python=${PYTHON_VERSION:-3.5} \
    && conda run -n ${CONDA_ENV} pip install --upgrade pip \
    && conda run -n ${CONDA_ENV} make install.requirements.development \
    && conda env export -n ${CONDA_ENV}
