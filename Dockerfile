FROM python:3.8.12-bullseye

WORKDIR /usr/src/app

LABEL Name=npanalyst Version=1.0.0
LABEL maintainer="jvansan <jeffreyavansanten@gmail.com>"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy npanalyst core lib
COPY setup.py ./
COPY ./npanalyst ./npanalyst
RUN pip install .

ENTRYPOINT [ "npanalyst" ]