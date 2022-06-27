"""
Module to provide processing functions for Audit Analytics data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np


class audit(wrds_module):
    """ Class providing main processing methods for the Audit Analytics data.

    One instance of this classs is automatically created and accessible from
    any data object instance.

    Args:
        w (pilates.data): Instance of pilates.data object.

    """

    def __init__(self, d):
        wrds_module.__init__(self, d)

    def get_fields_restatements(self, fields, data=None, lag=0):
        """ Returns fields from Audit Analytics 'auditnonreli' file.

        The 'auditnonreli' file is the non-reliance restatements file.
        """
        key = ['res_notif_key']
        # Open restatement data
        df = self.open_data(self.auditnonreli, key + fields)

        if data is not None:
            # Merge to user's data and return the requested fields
            dfin = pd.merge(data, df, how='left', on=key)
            dfin.index = data.index
            return(dfin[fields])
        else:
            return(df)
