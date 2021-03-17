"""
Module to provide processing functions for BHL data

BHL data refers to the Risk Aversion and Uncertainty series from
Bekaert, Hoerova and Lo Duca (2013).
http://mariehoerova.net/

"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pilates import data_module


class bhl(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)

    def add_file(self, name, path, force=False, types=None):
        """ Add a file to the module.
        Converts and make the file available for the module to use.

        Args:
            name (str): Name of the file to be used by the module.
                Should be 'bhl'
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
        filename_pq = self.w.datadir+filename+'.parquet'
        # Check if the file has already been converted
        if os.path.exists(filename_pq) and force is False:
            None
        else:
            # Open the file (BHL uses one excel file)
            df = pd.read_excel(path, sheet_name=1, usecols=[0,1,2])
            df.columns = ['date','uc','ra']
            # Need to correct the date field
            df['year'] = df.date.dt.year
            df['month'] = df.date.dt.month
            # Remove extra columns
            df = df.drop(['date'], axis=1)

            # Write the data
            t = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filename_pq, t.schema)
            pqwriter.write_table(t)
            pqwriter.close()
        # Link the name to the actual filename
        setattr(self, name, filename)

    def get_fields(self, data, fields):
        key = ['year', 'month']

        df = self.d.open_data(self.bhl, key+fields)
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
