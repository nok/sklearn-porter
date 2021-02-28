FROM continuumio/miniconda3:4.9.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        apt-utils \
        ca-certificates \
        software-properties-common \
        gnupg \
        make \
        sudo \
        wget && \
    wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | sudo apt-key add - && \
    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ && \
    apt-get update && \
    mkdir -p /usr/share/man/man1/ && \
    apt-get install -y --no-install-recommends \
        adoptopenjdk-8-hotspot \
        libopenblas-dev liblapack-dev gfortran cpp g++ gcc \
        ruby \
        php \
        nodejs \
        golang && \
    rm -rf /var/lib/apt/lists/*

RUN echo "\n\nJava version:\n" && java -version && \
    echo "\n\nGCC version:\n" && gcc --version && \
    echo "\n\nRuby version:\n" && ruby --version && \
    echo "\n\nPHP version:\n" && php --version && \
    echo "\n\nPython version:\n" && python --version && \
    echo "\n\nNode version:\n" && node --version && \
    echo "\n\nGo version:\n" && go version

ENV USER me
ENV HOME /home/${USER}
RUN mkdir -p ${HOME}/repo
COPY . ${HOME}/repo
WORKDIR ${HOME}/repo

ARG PYTHON_VER
ARG SKLEARN_VER
ARG EXTRAS

RUN conda config --set auto_activate_base true && \
    conda install -y -n base \
        ${PYTHON_VER:-python=3.6} \
        nomkl \
        cython numpy scipy \
        ${SKLEARN_VER:-scikit-learn} && \
    conda run -n base --no-capture-output python -m \
        pip install --no-cache-dir -U pip && \
    conda run -n base --no-capture-output python -m \
        pip install --no-cache-dir -e ".[${EXTRAS:-development,examples}]" && \
    conda clean --all -y && \
    conda run -n base --no-capture-output python -m \
        pip freeze | grep -i -E 'cython|numpy|scipy|scikit-learn'

RUN if [ -e /opt/conda/bin/ipython ] ; then \
        conda run --no-capture-output -n base \
            ipython profile create && \
        echo -e "\nc.InteractiveShellApp.exec_lines = [\x27import sys; sys.path.append(\x22${HOME}\x22)\x27]" >> $(conda run -n base ipython locate)/profile_default/ipython_config.py \
    ; fi

ENTRYPOINT ["./docker-entrypoint.sh"]
