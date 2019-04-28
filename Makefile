BASH := /bin/bash

#
# Requirements
#

install.environment:
	$(info Start [install.environment] ...)
	$(BASH) scripts/install.environment.sh

source.environment: install.environment
	$(info Start [source.environment] ...)
	$(BASH) scripts/source.environment.sh

install.requirements: source.environment
	$(info Start [install.requirements] ...)
	$(BASH) scripts/install.requirements.sh

install.requirements.examples: install.requirements
	$(info Start [install.requirements.examples] ...)
	$(BASH) scripts/install.requirements.examples.sh

install.requirements.development: install.requirements.examples
	$(info Start [install.requirements.development] ...)
	$(BASH) scripts/install.requirements.development.sh

#
# Examples
#

open.examples: install.requirements.examples examples.pid

examples.pid:
	$(info Start [examples.pid] ...)
	jupyter-lab --notebook-dir='examples' > /dev/null 2>&1 & echo $$! > $@;

stop.examples: examples.pid
	kill `cat $<` && rm $<

.PHONY: open.examples stop.examples

#
# Development
#

all: clean lint serve tests jupytext clean

lint: install.requirements.development
	$(info Start [lint] ...)
	$(eval FILES=$(shell find ./sklearn_porter -name '*.py' -type 'f' | tr '\n' ' '))
	pylint --rcfile=.pylintrc --output-format=text $(FILES) 2>&1 | tee pylint.txt | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p'

serve: source.environment
	$(info Start [serve] ...)
	$(BASH) scripts/run.server.sh

tests: local.tests

local.tests: install.requirements.development clean serve
	$(info Start [test] ...)
	$(BASH) scripts/run.tests.system.sh $(python_files)

docker.tests: clean
	$(info Start [docker.tests] ...)
	$(BASH) scripts/run.tests.docker.sh

jupytext: install.requirements.development
	$(info Start [jupytext] ...)
	$(BASH) scripts/run.jupytext.sh

deploy: install.requirements.development clean
	$(info Start [deploy.test] ...)
	$(BASH) scripts/run.deployment.sh

#
# Cleanup
#

clean: clean.build clean.pycache

clean.pycache:
	$(info Start [clean.pycache] ...)
	$(BASH) scripts/clean.pycache.sh

clean.build:
	$(info Start [clean.build] ...)
	$(BASH) scripts/clean.build.sh
