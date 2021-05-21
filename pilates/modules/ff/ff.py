"""
Module to provide processing functions for Fam-French data.

JLN data refers to the Economic Uncertainty data from
Jurado, Ludvigson and Ng (2015).
https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes

"""

import os, re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pilates import data_module


class ff(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        self.key = ['date']
        self.col_date = 'date'

    def add_file_definition(self, name):
        self._check_name(name)
        if not hasattr(self, name):
            filepath_pq = self._filepath_pq(name)
            # Check if the file has already been converted
            if not os.path.exists(filepath_pq):
                ## Path of the final file to convert
                path = self._path(name)
                if not os.path.exists(path):
                    self.download_file(name)
                # Convert the file
                ## Create a CSV file using Fama-French text file
                f = open(path)
                lines = f.readlines()
                f.close()
                df = pd.DataFrame(columns=['code', 'name_short', 'name', 'sic', 'name_sic'])
                i = 0
                while i<len(lines):
                    # Get the FF industry number and name
                    pattern = re.compile(r"\s*(\d*)\s*([a-zA-Z]*)\s*(\w.*)$")
                    res = pattern.search(lines[i])
                    num, ind, inddef = res.groups()
                    i += 1
                    # Get the SIC codes
                    while i<len(lines) and lines[i] != '\n':
                        pattern = re.compile(r"(\d\d\d\d)-(\d\d\d\d)\s*(\w?.*)$")
                        res = pattern.search(lines[i])
                        range_start, range_end, range_name = res.groups()
                        ran = range(int(range_start), int(range_end)+1)
                        obs = {}
                        for sic in ran:
                            obs['code'] = num
                            obs['name_short'] = ind
                            obs['name'] = inddef
                            obs['sic'] = sic
                            obs['name_sic'] = range_name
                            df = df.append(obs, ignore_index=True)
                        i += 1
                    i += 1 # Empty line
                # Save data as csv and change the path for conversion
                path = path.replace('.txt', '.csv')
                df.to_csv(path, index=False)
                ## Convert the CSV file using Pilates
                self._convert_data(name, path)
            # Add the file to the module
            setattr(self, name, name)

    def set_industry_definition(self, ind_def):
        """ Define the industry definition.
        Has to be either 5, 10, 12, 17, 30, 38, 48 or 49.
        """
        name = 'sic'+str(ind_def)
        self.add_file_definition(name)
        self.sic = getattr(self, name)

    def get_codes(self, data, fields=['code'], sic_col=None):
        """ Return the Fama-French industry code according to the chosen definition.
        The user must first set the definition with set_industry_definition().

        For example, to use FF 48 industries: set_industry_definition(48).

        The user data requires either the sic codes (column 'sic') or
        the gvkey and datadate columns from Compustat with access to compustat data.
        """
        if not hasattr(self, 'sic'):
            raise Exception('Please set the insdutry definition with'
                            ' set_industry_definition() first.')
        # Open the FF codes
        df = self.open_data(self.sic)
        # Merge to the user data
        dfu = data.copy()
        ## Make sure the sic code is available
        cols = dfu.columns
        if sic_col is not None:
            dfu['sic'] = dfu['sic_col']
        if 'sic' not in cols:
            if 'gvkey' not in cols or 'datadate' not in cols:
                raise Exception('Need either the sic information gvkey and datadate'
                                ' from Compustat.')
            # Retrieve SIC codes
            col_date = self.d.col_date
            col_id = self.d.col_id
            self.d.set_date_column('datadate')
            self.d.set_id_column('gvkey')
            dfu['sic'] = self.d.comp.get_fields(['sic'], data=dfu)
            self.d.set_date_column(col_date)
            self.d.set_id_column(col_id)
        ## Merge the FF codes
        dfin = pd.merge(dfu['sic'], df[['sic']+fields], how='left', on='sic')
        dfin.index = data.index
        return dfin[fields]
