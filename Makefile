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

install.requirements:
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

start.examples: install.requirements.examples examples.pid

examples.pid:
	$(info Start [examples.pid] ...)
	jupyter notebook --notebook-dir='examples/basics' --ip='0.0.0.0' --port=8888 > /dev/null 2>&1 & echo $$! > $@;

stop.examples: examples.pid
	kill `cat $<` && rm $<

.PHONY: open.examples stop.examples

#
# Development
#

all: clean lint tests examples clean

lint: install.requirements.development
	$(info Start [lint] ...)
	$(eval FILES=$(shell find ./sklearn_porter -name '*.py' -type f | tr '\n' ' '))
	pylint --rcfile=.pylintrc --output-format=text $(FILES) 2>&1 | tee pylint.txt | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p'

tests: tests.local

tests.local: install.requirements.development clean
	$(info Start [test.local] ...)
	$(BASH) scripts/run.tests.local.sh

tests.docker: clean
	$(info Start [docker.tests] ...)
	$(BASH) scripts/run.tests.docker.sh

examples: install.requirements.development
	$(info Start [jupytext] ...)
	$(BASH) scripts/run.jupytext.sh

deploy: install.requirements.development clean
	$(info Start [deploy] ...)
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
