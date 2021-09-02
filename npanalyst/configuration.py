import json
from copy import deepcopy
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Dict, Optional

from npanalyst.logging import get_logger
from npanalyst.msutils import make_error_col_names

logger = get_logger()

DEFAULT_CONFIG = {
    "MSFileInfo": {
        "FileNameCol": "UniqueFiles",
        "MS1Cols": "PrecMz,RetTime,PrecIntensity",
        "MS1ColsToMatch": "PrecMz,RetTime",
        "MSLevel": 1,
    },
    "Tolerances": {"PrecMz": "ppm,30", "RetTime": "window,0.03", "PrecZ": "None,None"},
    "ReplicateInfo": {"RequiredReplicates": 2, "MinimumIntensity": 2e3},
    "BasketInfo": {
        "ColumnsToFeature": "PrecMz,RetTime,PrecIntensity,UniqueFiles",
        "MinMaxCols": "PrecIntensity",
        "BasketMSLevel": 1,
        "CalcBasketInfo": False,
        "RequiredReplicates": 1,
    },
    "NetworkInfo": {"ActivityThreshold": 2, "ClusterThreshold": 0.3},
}


def load_raw_config(config_path: Optional[Path] = None) -> Dict:
    """Loads raw (overly-structures) config dictionary"""
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    try:
        with open(config_path) as f:
            return json.load(f)
    except OSError as e:
        logger.error("Could not find config file")
        raise e
    except JSONDecodeError as e:
        logger.error("Invalid JSON config file")
        raise e


def load_config(config_path: Optional[Path] = None) -> Dict:
    """loads the config_path config file and stores a bunch of values in a flatten dict
    config_path (str, optional): Defaults to 'default.json'.
        path to the config file, defaults can be overridden.
    """
    config = load_raw_config(config_path)
    MS1COLSTOMATCH = config["MSFileInfo"]["MS1ColsToMatch"].split(",")
    configd = {
        "FILENAMECOL": config["MSFileInfo"]["FileNameCol"],
        "MSLEVEL": int(config["MSFileInfo"]["MSLevel"]),
        "MS1COLS": config["MSFileInfo"]["MS1Cols"].split(","),
        "MS1COLSTOMATCH": MS1COLSTOMATCH,
        "MS1ERRORCOLS": make_error_col_names(MS1COLSTOMATCH),
        "CALCBASKETINFO": config["BasketInfo"]["CalcBasketInfo"],
        "BASKETMSLEVEL": int(config["BasketInfo"]["BasketMSLevel"]),
        "BASKETMINMAXCOLS": config["BasketInfo"]["MinMaxCols"].split(","),
        "BASKETFEATURES": config["BasketInfo"]["ColumnsToFeature"].split(","),
        "MINREPSBASKETS": int(config["BasketInfo"]["RequiredReplicates"]),
        "MINREPSREPLICATES": int(config["ReplicateInfo"]["RequiredReplicates"]),
        "MININTENSITY": int(config["ReplicateInfo"]["MinimumIntensity"]),
        "ACTIVITYTHRESHOLD": float(config["NetworkInfo"]["ActivityThreshold"]),
        "CLUSTERTHRESHOLD": float(config["NetworkInfo"]["ClusterThreshold"]),
    }

    ERRORINFO = {}
    for name, tup in config["Tolerances"].items():
        etype, ev = tup.split(",")
        if etype == "None":
            etype = None
        if ev == "None":
            ev = None
        else:
            ev = float(ev)
        ERRORINFO[name] = (etype, ev)
    configd["ERRORINFO"] = ERRORINFO

    logger.debug(f"Config loaded: \n{json.dumps(configd, indent=2)}")
    return configd
