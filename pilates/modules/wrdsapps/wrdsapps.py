"""
Module to provide processing functions for Compustat data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np


class wrdsapps(wrds_module):
    """ Class providing main processing methods for the wrdsapps data
    It mostly contains linktables.
    """

    def __init__(self, d):
        wrds_module.__init__(self, d)

    def infocode_from_code(self, data):
        """ Add Datastream infocode using worldscope code.
        Need worldscope code and a date (year or other).
        """
        # 1. Open Datastream - Worldscope linktable
        df = self.open_data(self.ds2ws_linktable,
                            columns=['code', 'infocode',
                                     'startdate', 'enddate',
                                     'inforank'])

        # 2. Add infocode to Worldscope code
        # Remember the user data index
        index = data.index

        # 2.1 Check the frequency used for worldscope
        if self.d.worldscope.freq == 'A':
            key = ['code', 'year_']
        elif self.d.worldscope.freq == 'Q':
            key = ['code', 'year_', 'seq']
        else:
            raise Exception('Need to define the wordlscope frequency used.')

        # 2.1 Merge the linktable to the user data
        dfm = data[key].dropna().merge(df, how='left', on='code') 
        dfm = dfm.dropna(subset=['infocode'])

        # Clean the merge using start and end dates of the links
        if self.d.worldscope.freq == 'A':
            dfm['date'] = pd.to_datetime(dfm.year_.astype(str) + '-12-31')
            dfm = dfm[(dfm.date >= dfm.startdate) & (dfm.date <= dfm.enddate)]
            # Keep the lowest inforank when duplicates of [code, year_]
            dfmg = dfm.groupby(['code', 'year_'])
            dfm['inforank_min'] = dfmg['inforank'].transform(min)
            dfm = dfm[dfm.inforank == dfm.inforank_min]
        elif self.d.worldscope.freq == 'Q':
            raise Exception('Adding Datastream infocode to quarterly'
                            'worldscope is not yet supported.')

        # Note: it is a n-n mapping! 
        dfm = dfm[key + ['infocode']]
        dfin = data[key].merge(dfm, how='left', on=key)
        return(dfin.infocode)