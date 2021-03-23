"""
Module to provide processing functions for BEX data.

BEX data refers to the Economic Uncertainty data from
Bekaert, Engstrom and Xu (2020).
https://www.nancyxu.net/risk-aversion-index

"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pilates import data_module


class bex(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        self.key = ['date']
        self.col_date = 'date'

    def _convert_data(self, name, path,
                     force=False,
                     delim_whitespace=False,
                     nrows=None):
        # Open the file
        sheet_name = self.files[name]['sheet']
        df = pd.read_excel(path, sheet_name=sheet_name, usecols=[0,1,2,3])
        # Correct the date field
        if 'yyyymm' in df.columns:
            df['date'] = pd.to_datetime(df.yyyymm, format='%Y%m')
            df = df.drop(['yyyymm'], axis=1)
        elif 'yyyymmdd' in df.columns:
            df['date'] = pd.to_datetime(df.yyyymmdd, format='%Y%m%d')
            df = df.drop(['yyyymmdd'], axis=1)
        # Write the data
        self._write_to_parquet(df, name)

    def set_frequency(self, frequency):
        """ Define the frequency of the indices.
        It can be 'monthly' or 'daily'
        """
        if frequency in ['monthly', 'daily']:
            self.add_file(frequency)
            self.bexfile = frequency
            self.frequency = frequency
        else:
            raise Exception('BEX frequency should by either',
                            'monthly or daily')

    def get_fields(self, data, fields):
        df = self.open_data(self.bexfile, self.key+fields)
        # User data
        dfu = self.open_data(data, [self.d.col_date])
        # Merge
        dfu = dfu.sort_values(self.d.col_date)
        df = df.sort_values(self.col_date)
        if self.frequency == 'monthly':
            tolerance = pd.Timedelta('31 day')
        elif self.freqency == 'daily':
            tolerance =pd.Timedelta('6 day')
        dfin = pd.merge_asof(dfu, df,
                             left_on=self.d.col_date,
                             right_on=self.col_date,
                             tolerance=tolerance,
                             direction='nearest')
        dfin.index = dfu.index
        return(dfin[fields])
