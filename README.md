# metabolate

Tool-chain for relating metabolite MS data with bio-activity data. 

## Rtree Based Basketing

Using an rtree to build connected component graphs of mz features with overlapping error ranges (in all dimensions). These are then combined (averaged) and represent a replicated or basketed feature.

## Functionality

Right now things are implemented via a CLI (`cli.py`) which will process a folder of csv files or basket a folder of replicated CSV's (output from replicate task). This will serve as the web backend for the flask app. 
