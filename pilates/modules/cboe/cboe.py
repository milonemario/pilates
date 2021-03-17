"""
Provides processing functions for the CBOE data.
"""

from pilates import data_module


class cboe(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        # Initialize values
        self.col_date = 'date'

    def get_fields(self, data, fields):
        key = ['date']
        df = self.d.open_data(self.cboe, key+fields)
        # Merge to the use data
        dfu = self.d.open_data(data, [self.d.col_date])
        dfu['date'] = dfu[self.d.col_date]
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])
