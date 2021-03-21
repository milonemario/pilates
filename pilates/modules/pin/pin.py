"""
Module to provide processing functions for PIN data.

PIN data refers to the Probability of Informed Trading from
Easley et al. (2002).

Data vailable here:
https://scholar.rhsmith.umd.edu/sbrown/pin-data

"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pilates import data_module


class pin(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        # Initialize values
        self.col_id = 'permno'
        self.col_date = 'year'
        self.key = [self.col_id, self.col_date]
        self.freq = None  # Data frequency

    def set_frequency(self, frequency):
        """ Define the frequency to use for the PIN data.
        The corresponding 'pins_vdj' file (ann or qtr)
        must be added to the module before setting the frequency.

        Args:
            frequency (str): Frequency of the data.
                Supports 'Quarterly' and 'Annual'.

        Raises:
            Exception: If the frequency is not supported.

        """
        if frequency in ['Quarterly', 'quarterly', 'Q', 'q']:
            self.freq = 'Q'
            self.add_file('qtr')
            self.pins = self.qtr
        elif frequency in ['Annual', 'annual', 'A', 'a',
        'Yearly', 'yearly', 'Y', 'y']:
            self.freq = 'A'
            self.add_file('ann')
            self.pins = self.ann
        else:
            raise Exception('PIN data frequency should by either '
                            'Annual or Quarterly')

    def get_fields(self, data, fields):
        key = self.key
        df = self.open_data(self.pins, key+fields)
        # Merge to the user data
        dfu = self.open_data(data, [self.col_id, self.d.col_date])
        # Extract year from user data
        if self.d.col_date_type == 'date':
            dt = pd.to_datetime(dfu[self.d.col_date]).dt
            dfu[self.col_daate] = dt.year
        elif self.d.col_date_type == 'year':
            dfu[self.col_date] = dfu[self.d.col_date]
        # Merge
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])
