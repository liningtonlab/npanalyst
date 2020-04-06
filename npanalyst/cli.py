from pathlib import Path
from npanalyst import core


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        help="task to perform.",
        choices=["replicate", "basket", "both", "activity", "full_pipeline"],
    )
    parser.add_argument(
        "path",
        help="path to input for either task or basketed data for activity mapping",
    )
    parser.add_argument("-o", "--output", help="Output directory", default=".")
    parser.add_argument(
        "-w",
        "--workers",
        help="number of parallel workers to spin up",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-f", "--filename_col", help="column name for the filename", default="Sample"
    )
    parser.add_argument(
        "--basket_info",
        help="Flag to save basket info as a json object in resulting files.",
        action="store_true",
    )
    parser.add_argument(
        "--ms2",
        help="match ms2 ions during basketing Note: Will drastically increase time required",
        action="store_true",
    )
    parser.add_argument(
        "--activity_data",
        help="path to activity data or folder containing multiple activity files",
    )
    parser.add_argument(
        "--config",
        help="custom config file to use. arguments will overwrite an overlapping config file options",
    )
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    args = parser.parse_args()

    core.setup_logging(args.verbose)

    # Check required fields are satisfied based on arguments
    if args.task in ("full_pipeline", "activity") and not args.activity_data:
        parser.error("Activity data is missing")
    if args.task == "activity" and "Basketed.csv" not in args.path:
        parser.error("Path argument must be to basketed data file")

    if args.config:
        configd = core.load_config(args.config)
    else:
        configd = core.load_config()

    if args.filename_col:
        configd["FILENAMECOL"] = args.filename_col

    data_path = Path(args.path)
    if args.task in ["replicate", "both", "full_pipeline"]:
        core.mp_proc_folder(data_path, configd, max_workers=args.workers)

    if args.task in ["basket", "both", "full_pipeline"]:
        if args.task in ["both", "full_pipeline"]:
            data_path = data_path.joinpath("Replicated")
        else:
            data_path = Path(args.path)
        core.basket(data_path, configd)
    if args.activity_data and args.task in ["activity", "full_pipeline"]:
        outdir = Path(args.output)
        if not outdir.exists():
            outdir.mkdir()
        configd["outputdir"] = outdir.resolve()
        if args.task == "full_pipeline":
            basket_path = (
                Path(args.path).joinpath("Replicated").joinpath("Basketed.csv")
            )
        else:
            basket_path = Path(args.path)
        core.load_and_generate_act_outputs(basket_path, args.activity_data, configd)


if __name__ == "__main__":
    main()
