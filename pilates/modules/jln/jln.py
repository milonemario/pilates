"""
Module to provide processing functions for JLN data.

JLN data refers to the Economic Uncertainty data from
Jurado, Ludvigson and Ng (2015).
https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes

"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pilates import data_module


class jln(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        self.key = ['date']
        self.col_date = 'date'

    def set_type(self, type):
        """ Define the uncertainty type to use.
        It can be 'financial', 'macro', or 'real'
        """
        if type in self.files.keys():
            self.add_file(type)
            self.jlnfile = getattr(self, type)
        else:
            raise Exception('JLN uncertainty type should by either',
                            self.files.keys())

    def get_fields(self, data, fields):
        df = self.open_data(self.jlnfile, self.key+fields)
        # Merge to the user data
        dfu = self.open_data(data, [self.d.col_date])
        # Merge
        dfu = dfu.sort_values(self.d.col_date)
        df = df.sort_values(self.col_date)
        dfin = pd.merge_asof(dfu, df,
                             left_on=self.d.col_date,
                             right_on=self.col_date,
                             tolerance=pd.Timedelta('90 day'),
                             direction='nearest')
        dfin.index = dfu.index
        return(dfin[fields])
