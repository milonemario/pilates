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
        self.key = ['date']
        self.col_date = 'date'

    def add_file_old(self, name, path, force=False, types=None):
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

    def select_source(self, source):
        if source in self.files.keys():
            self.add_file(source)
            self.bhl = getattr(self, source)
        else:
            raise Exception("The available sources for BHL data "
                            "are "+str(self.files.keys()))

    def get_fields(self, data, fields):
        df = self.open_data(self.bhl, self.key+fields)
        # User data
        dfu = self.open_data(data, [self.d.col_date])
        # Merge
        dfu = dfu.sort_values(self.d.col_date)
        df = df.sort_values(self.col_date)
        dfin = pd.merge_asof(dfu, df,
                             left_on=self.d.col_date,
                             right_on=self.col_date,
                             tolerance=pd.Timedelta('31 day'),
                             direction='nearest')
        dfin.index = dfu.index
        return(dfin[fields])
