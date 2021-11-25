# npanalyst

Tool-chain for relating metabolite MS data with bio-activity data.

For complete usage of the CLI [see our documentation](https://liningtonlab.github.io/npanalyst_documentation/NPAnalyst/cli-tutorial/).

## Installation

### Local

Using Anaconda Python, create a virtual environment with all the necessary
dependencies, and then install the toolchain into that environment:

```
conda env create -f environment.yml
conda activate npanalyst
pip install .
```

### Docker

You can use [Docker](https://www.docker.com/get-started) to run the `npanalyst` CLI by binding a local volume.

To get the Docker image, you must build it. 

```bash
docker build -t npanalyst_cli .
```

Then running the CLI works as

See the CLI help command

```bash
docker run -it npanalyst_cli 
```

As an example of bind mounting a local volume.

```bash
docker run -it -v PATH/TO/DATA:/data npanalyst_cli import -i /data/GNPS.graphml -o /data -t GNPS -v
```

#### 

## Rtree Based Basketing

Using an rtree to build connected component graphs of mz features with overlapping error ranges (in all dimensions). 
These are then combined (averaged) and represent a replicated or basketed feature.

## Functionality

Right now things are implemented via a CLI (`npanalyst`) which will process a folder of csv files or basket a folder 
of replicated CSV's (output from replicate task).

`npanalyst --help` should show all the features and usage.

A lot of the parameters are specified via a config file that defaults to `npanalyst.configuration.DEFAULT_CONFIG` 
this will change a little as things get tuned. You can generate a config file using `npanalyst get_config` which will 
produce a `./config.json` file.


## Tests

Pytest has been used to implement basic unit and full pipeline (called integration here) tests.

To run the tests, make sure you have pytest installed (Eg. `pip install pytest`), and simply run the `pytest` command.
You can run the unit and integration tests separately by specifying a path to test `pytest tests/unit` or `pytest tests/integration`.