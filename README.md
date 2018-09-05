# metabolate

Tool-chain for relating metabolite MS data with bio-activity data.

## Rtree Based Basketing

Using an rtree to build connected component graphs of mz features with overlapping error ranges (in all dimensions). These are then combined (averaged) and represent a replicated or basketed feature.

## Functionality

Right now things are implemented via a CLI (`metabolate`) which will process a folder of csv files or basket a folder of replicated CSV's (output from replicate task). 

A lot of the parameters are specified via a config file that defaults to `default.cfg` this will change a little as things get tuned. 

Additionally activity mapping is roughly implemented now focused on generating inputs for the bokeh server and cytoscape files. Additional options and work to be done...

## ToDo

* Add functionality to config file
* Polish Activity mapping
  
  * more agnostic on input type
  * add threshold parameters to config files

* Alternative visualization and outputs
