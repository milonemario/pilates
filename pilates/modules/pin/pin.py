"""
Module to provide processing functions for PIN data.

PIN data refers to the Probability of Informed Trading from
Easley et al. (2002).
http://www.smith.umd.edu/faculty/hvidkjaer/data.htm

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
            # Open the file (PIN uses ASC file (tab separated))
            df = pd.read_csv(path, delim_whitespace=True, nrows=nrows)
            df.columns = map(str.lower, df.columns)  # Lower case col names
            # Write the data
            t = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filename_pq, t.schema)
            pqwriter.write_table(t)
            pqwriter.close()
        # Link the name to the actual filename
        setattr(self, name, filename)

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
            self.pins = self.qtr
        elif frequency in ['Annual', 'annual', 'A', 'a',
        'Yearly', 'yearly', 'Y', 'y']:
            self.freq = 'A'
            self.pins= self.ann
        else:
            raise Exception('PIN data frequency should by either',
                            'Annual or Quarterly')

    def get_fields(self, data, fields):
        key = self.key
        df = self.d.open_data(self.pins, key+fields)
        # Merge to the user data
        dfu = self.d.open_data(data, [self.col_id, self.d.col_date])
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
