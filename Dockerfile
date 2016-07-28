FROM andrewosh/binder-base

MAINTAINER Fabian Rost <fabian.rost@tu-dresden.de>

USER main

RUN conda create -q -n python2 python=2 anaconda pymc
RUN /bin/bash -c "source activate python2 && ipython kernel install --user"

RUN pip install --upgrade pip

RUN pip install uncertainties ipycache iminuit probfit

ENV SHELL /bin/bash
