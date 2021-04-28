"""
Module to provide processing functions for ISS data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import os, zipfile, re
from codecs import open


class iss(wrds_module):
    """ Class providing main processing methods for the ISS data.

    One instance of this classs is automatically created and accessible from
    any data object instance.

    Args:
        d (pilates.data): Instance of pilates.data object.

    """

    def __init__(self, d):
        wrds_module.__init__(self, d)
