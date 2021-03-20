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
        self.type = 'financial'

    def add_file(self, name, path, nrows=None, force=False, types=None):
        """ Add a file to the module.
        Converts and make the file available for the module to use.

        Args:
            name (str): Name of the file to be used by the module.
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
        filename_pq = self.d.datadir+filename+'.parquet'
        # Check if the file has already been converted
        if os.path.exists(filename_pq) and force is False:
            None
        else:
            # Open the file (EPU uses excel files)
            df = pd.read_csv(path, nrows=nrows)
            df.columns = map(str.lower, df.columns)  # Lower case col names
            # Need to correct the date field
            if name == 'financial':
                dfdate = pd.to_datetime(df.date, format='%m/%Y')
                df['year'] = dfdate.dt.year
                df['month'] = dfdate.dt.month
                # Remove extra columns
                df = df.drop(['date'], axis=1)
            elif name in ['macro', 'real']:
                dfdate = df.date.str.split('-', expand=True)
                dfmonth = dfdate[0]
                df['month'] = pd.to_datetime(dfmonth, format='%b').dt.month
                dfyear = dfdate[1].astype(int)
                # Assume that there is no data after 2030
                df.loc[dfyear<30, 'century'] = int(20)
                df.loc[dfyear>30, 'century'] = 19
                df['year'] = (100*df.century + dfyear).astype(int)
                # Remove extra columns
                df = df.drop(['date', 'century'], axis=1)

            # Write the data
            t = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filename_pq, t.schema)
            pqwriter.write_table(t)
            pqwriter.close()
        # Link the name to the actual filename
        setattr(self, name, filename)

    def set_type(self, type):
        """ Define the uncertainty type to use.
        It can be 'financial', 'macro', or 'real'
        """
        types = ['financial', 'macro', 'real']
        if type in types:
            self.type = type
        else:
            raise Exception('JLN uncertainty type should by either',
                            types)

    def get_fields(self, data, fields):
        key = ['year', 'month']
        jlnfile = getattr(self, self.type)

        df = self.open_data(jlnfile, key+fields)
        # Merge to the user data
        dfu = self.open_data(data, [self.d.col_date])
        # Extract year and month from user data
        dt = pd.to_datetime(dfu[self.d.col_date]).dt
        dfu['year'] = dt.year
        dfu['month'] = dt.month
        # Merge
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])
