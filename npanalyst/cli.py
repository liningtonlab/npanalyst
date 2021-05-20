from pathlib import Path
import sys

from npanalyst import core, utils, activity
from npanalyst.config import load_config
from npanalyst.convert import mzmine, gnps, default

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        help="task to perform.",
        choices=["replicate", "basket", "activity", "full_pipeline", "import"],
    )
    parser.add_argument(
        "path",
        help="path to input for either task or basketed data for activity mapping",
        type=str,
    )
    parser.add_argument(
        "--msdatatype",
        choices=["mzml", "mzmine", "gnps"],
        default="mzml",
        help="types of inputs inputs accepted: mzml (default), mzmine, and gnps"
    )
    parser.add_argument("-o", "--output", help="Output directory", default=".")
    parser.add_argument(
        "-w",
        "--workers",
        help="number of parallel workers to spin up",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-f", "--filename_col", help="column name for the filename", default="Sample"
    )
    parser.add_argument(
        "--activity_data",
        help="path to activity data or folder containing multiple activity files.",
    )
    parser.add_argument(
        "--config",
        help="custom config file to use. arguments will overwrite an overlapping config file options",
    )
    parser.add_argument(
        "--activity_threshold",
        help="Activity threshold, default=3",
        default=3,
    )
    parser.add_argument(
        "--cluster_threshold",
        help="Cluster threshold, default=0",
        default=0,
    )
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    args, unknown = parser.parse_known_args()

    # Checking to see if all the files listed using the glob operator "*" are of the same extension
    if (unknown): 
        if not (utils.sameFileFormat(args.path, unknown)):
            parser.error("All imported files must have the same extension")
        data_path = Path(args.path).parent
    else:
        data_path = Path(args.path)

    baseDir = Path(args.path)

    core.setup_logging(args.verbose)

    if args.msdatatype not in ["mzml", "mzmine", "gnps"]:
        parser.error("Only mzml, mzmine and gnps filetypes accepted")
    else:
        msdatatype = args.msdatatype
    
    # Check required fields are satisfied based on arguments
    if args.task in ("full_pipeline", "activity", "import") and not args.activity_data:
        parser.error("Activity data is missing")
        
    if args.config:
        configd = core.load_config(args.config)
    else:
        configd = core.load_config()

    # load entire function call
    configd["CALL"] = vars(args)

    if args.filename_col:
        configd["FILENAMECOL"] = args.filename_col

    clustThreshold = str(args.cluster_threshold)
    if not clustThreshold == "auto": 
        if (abs(float(clustThreshold)) > 1):
            parser.error("Cluster threshold must be between -1 and 1")
        else:
            configd["CLUSTERTHRESHOLD"] = float(clustThreshold)
            print("CLUSTERTHRESHOLD set to", clustThreshold)
    else:
        configd["CLUSTERTHRESHOLD"] = "auto"
        print ("Autodetect cluster threshold enabled")      # not developed yet
    
    actTreshold = str(args.activity_threshold)
    if not actTreshold == "auto":
        if (float(actTreshold) < 0):
            parser.error("Activity threshold can not be negative")
        else:
            configd["ACTIVITYTHRESHOLD"] = float(actTreshold)
            print("ACTIVITYTHRESHOLD set to", actTreshold)
    else:
        configd["ACTIVITYTHRESHOLD"] = "auto"
        print ("Autodetect activity threshold enabled") # not developed yet

    # store the msdatatype - should be configured to be read in REACT
    configd["MSDATATYPE"] = msdatatype

    # save the config file
    if (msdatatype != "mzml"):
        baseDir = baseDir.parent.parent.parent
    else:
        baseDir = baseDir.parent.parent
    # core.save_config(baseDir, configd)

    output_path = Path(args.output)
    if not args.output == ".":
        if not (output_path.exists()):
            print ("Output path does not exist")

    # Check to see if the activity file exists
    if not (Path(args.activity_data).exists()):
        parser.error("The activity file does not exist.")
    else:
        act_path = args.activity_data
        # load sample names to be used for entire analysis - taken from activity_data
        samples = utils.get_samples(act_path, configd["FILENAMECOL"])

    if not (data_path.exists()):
        parser.error("The data path/files do not exist.")
    
    if msdatatype == "mzml":
        print ("Running mzml.")

        try:
            # consider full_pipeline to be the same as import for mzml
            if args.task == "import":   
                args.task = "full_pipeline"

            if args.task in ["activity"] and "basketed.csv" not in args.path:
                parser.error("Path argument must be to basketed data file")
            
            if args.task in ["replicate", "full_pipeline"]:
                core.proc_folder(data_path, output_path, configd, msdatatype, samples, max_workers=args.workers)

            if args.task in ["basket", "full_pipeline"]:
                if args.task in "full_pipeline":
                    #data_path = data_path.joinpath("Replicated")
                    output_path = output_path.joinpath("replicated")
                    core.basket(output_path, configd)
                else:
                    #data_path = Path(args.path)
                    core.basket(data_path, configd)
                #core.basket(data_path, configd)

            if args.activity_data and args.task in ["activity", "full_pipeline"]:
                outdir = Path(args.output)
                if not outdir.exists():
                    outdir.mkdir()
                configd["OUTPUTDIR"] = outdir.resolve()
                    
                if args.task == "full_pipeline":
                    if (output_path.exists()):
                        basket_path = output_path.joinpath("basketed.csv")
                    else:
                        basket_path = data_path.joinpath("replicated").joinpath("basketed.csv")
                else:
                    basket_path = Path(args.path)

                core.load_and_generate_act_outputs(basket_path, args.activity_data, configd)
                # core.create_clusters(act_path, configd["OUTPUTDIR"])
                configd['MAXCLUSTERS'] = core.create_communitites(act_path, configd["OUTPUTDIR"])
                core.save_config(baseDir, configd)
                print ("Mzml analysis completed.")
        except: 
            print ("Mzml analysis failed")
        

    elif msdatatype == "mzmine":
        print ("Running mzmine converter.")
        
        outdir = Path(args.output)
        if not outdir.exists():
            outdir.mkdir()
        configd["OUTPUTDIR"] = outdir.resolve()

        # convert into mzmine
        try:
            basket_path = configd["OUTPUTDIR"].joinpath("basketed.csv")
            default(act_path, data_path, configd)
            # mzmine(act_path, data_path, configd)
            core.load_and_generate_act_outputs(basket_path, act_path, configd)
            # core.create_clusters(act_path, configd["OUTPUTDIR"])
            configd['MAXCLUSTERS'] = core.create_communitites(act_path, configd["OUTPUTDIR"])
            core.save_config(baseDir, configd)
            print ("Mzmine conversion completed.")
        except:
            print ("Mzmine conversion failed")


    elif msdatatype == "gnps":
        print ("Running gnps converter.")

        outdir = Path(args.output)
        if not outdir.exists():
            outdir.mkdir()
        configd["OUTPUTDIR"] = outdir.resolve()

        try:
            basket_path = configd["OUTPUTDIR"].joinpath("basketed.csv")
            gnps(act_path, data_path, configd)
            core.load_and_generate_act_outputs(basket_path, args.activity_data, configd)
            # core.create_clusters(act_path, configd["OUTPUTDIR"])
            configd['MAXCLUSTERS'] = core.create_communitites(act_path, configd["OUTPUTDIR"])
            core.save_config(baseDir, configd)
            print ("GNPS conversion completed.")
        except:
            print ("GNPS conversion failed")


if __name__ == "__main__":
    main()
