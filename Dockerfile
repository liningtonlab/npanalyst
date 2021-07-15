FROM continuumio/miniconda3
LABEL maintainer="jvansan <jeffreyavansanten@gmail.com>"

WORKDIR /usr/src/app

# Copy npanalyst CLI and install
COPY . /usr/src/app
RUN conda env create -f environment.yml
RUN conda run -n npanalyst pip install . --use-feature=in-tree-build

ENTRYPOINT [ "/opt/conda/envs/npanalyst/bin/npanalyst"]