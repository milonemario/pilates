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
        self.frequency = 'monthly'

    def add_file(self, name, path, force=False, types=None):
        """ Add a file to the module.
        Converts and make the file available for the module to use.

        Args:
            name (str): Name of the file to be used by the module.
                Should be 'monthly' or 'daily'
            path (str): Path of the original file to convert.
            force (bool, optional): If False, the file is not re-converted.
                Defaults to False.
            types (str, optional): Path of the Yaml file containing the fields
                types of the data.

        """

        self.d._check_data_dir()
        # Get the file type
        filename, ext = os.path.splitext(os.path.basename(path))
        # Create the new file name
        filename_pq = self.d.datadir+filename+name+'.parquet'
        # Check if the file has already been converted
        if os.path.exists(filename_pq) and force is False:
            None
        else:
            # Open the file (BEX uses one excel file)
            if name in ['monthly', 'daily']:
                sheet_name = 'DATA_PLOT_'+name
            else:
                raise Exception('BEX name should by either',
                                'monthly or daily')

            df = pd.read_excel(path, sheet_name=sheet_name, usecols=[0,1,2,3])
            # Write the data
            df.columns = map(str.lower, df.columns)  # Lower case col names
            # Need to correct the date field
            dfdate = pd.to_datetime(df.yyyymm, format='%Y%m')
            df['year'] = dfdate.dt.year
            df['month'] = dfdate.dt.month
            # Remove extra columns
            df = df.drop(['yyyymm'], axis=1)

            t = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filename_pq, t.schema)
            pqwriter.write_table(t)
            pqwriter.close()
        # Link the name to the actual filename
        setattr(self, name, filename+name)

    def set_frequency(self, frequency):
        """ Define the frequency of the indices.
        It can be 'monthly' or 'daily'
        """
        if freqency in ['monthly', 'daily']:
            self.frequency = frequency
        else:
            raise Exception('BEX frequency should by either',
                            'monthly or daily')

    def get_fields(self, data, fields):
        key = ['year', 'month']
        bexfile = getattr(self, self.frequency)

        df = self.d.open_data(bexfile, key+fields)
        # Merge to the user data
        dfu = self.d.open_data(data, [self.d.col_date])
        # Extract year and month from user data
        dt = pd.to_datetime(dfu[self.d.col_date]).dt
        dfu['year'] = dt.year
        dfu['month'] = dt.month
        # Merge
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])
