from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union, Optional


class Config:
    """Default configuration for running npanalyst
    """
    filename_col: str = "Sample"

    # Replicate
    rep_required_replicates: int = 2
    ms_level: int = 1
    ms1_cols: List[str] = ["PrecMz", "RetTime", "PrecIntensity"]
    ms1_cols_match: List[str]  = ["PrecMz", "RetTime"]
    ms1_tolerances: Dict[str, Tuple] = {
        "PrecMz": ("ppm", 30.0),
        "RetTime": ("window", 0.03),

    }
    ## MS2
    ms2_cols: List[str] =  ["ProdMz", "ProdIntensity", "Ar1", "Ar3"]
    ms2_cols_match: List[str] =  ["ProdMz", "Ar1"]
    ms2_tolerances: Dict[str, Tuple] = {
        "ProdMz": ("ppm", 25.0),
        "Ar1": ("window", 0.33),
    }
    # Basket
    basket_required_replicates: int = 1
    calc_basket_info: bool = True

    # Activity
    activity_threshold: float  = 5.0
    cluster_threshold: float = 0.25

    def _attr(self) -> Iterable:
        return iter(
            key for key in dir(self)
            if not callable(getattr(self, key)) and not key.startswith("__")
        )

    def to_dict(self) -> Dict:
        """Return dictionary of configuration

        Returns:
            Dict: key, value map of configuration
        """
        return dict(
            (key, getattr(self, key)) for key in self._attr()
        )


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    return Config()

def change_setting(self, attr, value) -> Dict:
    """Return updated dictionary of the configuration file

    Returns: 
        Dict: key, value map of configuration updated with the attr and value provided
    """
    #print (self, attr, value)
    config = self

    for key in self:
        if (key == attr):
            print ("Changing settings in", key, "to", value)
            config.update({key: value})
            print ("Changed settings")
    return config

#newconfig = Config().to_dict()
#print(change_setting(newconfig, "activity_threshold", 10))
#print(change_setting(newconfig, "cluster_threshold", 10))