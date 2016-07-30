FROM andrewosh/binder-base

MAINTAINER Fabian Rost <fabian.rost@tu-dresden.de>

USER main

RUN pip install --upgrade pip

RUN pip install uncertainties ipycache iminuit probfit

ENV SHELL /bin/bash
