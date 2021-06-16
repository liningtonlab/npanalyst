# npanalyst

Tool-chain for relating metabolite MS data with bio-activity data.

## Installation

Using Anaconda Python, create a virtual environment with all the necessary
dependencies, and then install the toolchain into that environment:

```
conda env create -f environment.yml
conda activate npanalyst
pip install .
```

## Rtree Based Basketing

Using an rtree to build connected component graphs of mz features with overlapping error ranges (in all dimensions). These are then combined (averaged) and represent a replicated or basketed feature.

## Functionality

Right now things are implemented via a CLI (`npanalyst`) which will process a folder of csv files or basket a folder of replicated CSV's (output from replicate task).

`npanalyst --help` should show all the features and usage.

A lot of the parameters are specified via a config file that defaults to `default.json` this will change a little as things get tuned.

Additional options and work to be done...

## TODO

- Add functionality to config file
- Polish Activity mapping

  - more agnostic on input type
  - add threshold parameters to config files

### Unit Tests

Use pytest to implement as much unit test coverage as possible.
