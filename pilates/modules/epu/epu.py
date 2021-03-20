"""
Module to provide processing functions for EPU data.

EPU data refers to the Economic Policy Uncertainty data from
Baker, Bloom and Davis (2016).
https://www.policyuncertainty.com

"""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pilates import data_module


class epu(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        self.level = 'country'
        self.maindata = 'comp'

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
            df = pd.read_excel(path, nrows=nrows)

            # Write the data
            df.columns = map(str.lower, df.columns)  # Lower case col names
            t = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filename_pq, t.schema)
            pqwriter.write_table(t)
            pqwriter.close()
        # Link the name to the actual filename
        setattr(self, name, filename)

    def set_level(self, level):
        """ Define the aggregation level of the index.
        It can be 'country' or 'firm'
        """
        if level in ['country', 'firm']:
            self.level = level
        else:
            raise Exception('EPU aggregation level should by either',
                            'country or firm')

    def set_maindata(self, maindata):
        """ Set the main data used in the study (for merging purposes).
        Could be 'comp' for compustat of 'crps' for CRSP.
        """
        if maindata in ['comp', 'crsp']:
            self.maindata = maindata
        else:
            raise Exception('EPU main data should by either',
                            'comp or crsp')

    def get_fields(self, data, fields):
        if self.level=='country':
            epuname = self.epu_country
            key = ['year', 'month']
            data_key = [self.d.col_date]
        elif self.level=='firm':
            epuname = self.epu_firm
            if self.maindata == 'comp':
                key = ['gvkey', 'year']
                data_key = ['gvkey', self.d.col_date]
            elif self.maindata == 'crsp':
                key = ['permno', 'year']
                data_key = ['permno', self.d.col_date]

        df = self.open_data(epuname, key+fields)
        if self.level == 'firm':
            # 2 duplicates on that file at locs [29196, 52509]
            df = df.drop([29196, 52509])
        # Merge to the user data
        dfu = self.open_data(data, data_key)
        # Extract year and month from user data
        dt = pd.to_datetime(dfu[self.d.col_date]).dt
        dfu['year'] = dt.year
        dfu['month'] = dt.month
        # Merge
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])
