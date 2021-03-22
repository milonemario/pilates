"""
Provides processing functions for the CBOE data from WRDS.
"""

from pilates import wrds_module


class cboe(wrds_module):

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_date = 'date'

    def get_fields(self, data, fields):
        key = ['date']
        df = self.open_data(self.cboe, key+fields)
        # Merge to the use data
        dfu = self.open_data(data, [self.d.col_date])
        dfu['date'] = dfu[self.d.col_date]
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])
