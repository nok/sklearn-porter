FROM continuumio/miniconda3:4.11.0

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

ENV USER user
ENV HOME /home/${USER}
ENV CONDA_AUTO_ACTIVATE_BASE true
ENV CONDA_ALWAYS_YES true
ENV CONDA_AUTO_UPDATE_CONDA false

RUN mkdir -p ${HOME}/repo
COPY . ${HOME}/repo
WORKDIR ${HOME}/repo

ARG PYTHON_VER
ARG SKLEARN_VER
ARG EXTRAS

RUN conda create -n sklearn-porter ${PYTHON_VER:-python=3.6} && \
    conda run -n sklearn-porter --no-capture-output python -m \
        pip install --no-cache-dir -U pip && \
    conda run -n sklearn-porter --no-capture-output python -m \
        pip install --no-cache-dir \
            -e ".[${EXTRAS:-development,examples}]" \
            "${SKLEARN_VER:-scikit-learn}" \
            cython numpy scipy && \
    conda clean --all && \
    conda run -n sklearn-porter --no-capture-output python -m \
        pip freeze | grep -i -E 'cython|numpy|scipy|scikit-learn'

RUN if [ -e /opt/conda/bin/ipython ] ; then \
        conda run --no-capture-output -n sklearn-porter \
            ipython profile create && \
        echo -e "\nc.InteractiveShellApp.exec_lines = [\x27import sys; sys.path.append(\x22${HOME}\x22)\x27]" >> $(conda run -n base ipython locate)/profile_default/ipython_config.py \
    ; fi

ENTRYPOINT ["./docker-entrypoint.sh"]
