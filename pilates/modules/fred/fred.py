"""
Provide convenient functions to get FRED economic data.
"""

from pilates import data_module
import pandas as pd
from fredapi import Fred


class fred(data_module):

    def __init__(self, w):
        data_module.__init__(self, w)

    def set_api_key(self, api_key):
        self.fred = Fred(api_key=api_key)

    def get_10y_US_rates(self, data, col_date=None):
        """ Return the 10 years treasury rates.
        """
        if col_date is None:
            col_date = self.d.col_date
        df_fred = self.fred.get_series('DGS10')
        df = pd.DataFrame(df_fred).reset_index()
        df.columns = ['date', 'r']
        dfu = self.open_data(data, [col_date])
        dfu.columns = ['date']
        # If quarterly data, use the last day of the month
        if self.d.freq == 'Q':
            df['y'] = df.date.dt.year
            df['m'] = df.date.dt.month
            df['month_end'] = df.groupby(['y', 'm']).date.transform('max')
            df = df[df.date == df.month_end]
            dt = pd.to_datetime(dfu.date).dt
            dfu['y'] = dt.year
            dfu['m'] = dt.month
            key = ['y', 'm']
        else:
            key = 'date'
        dfin = dfu.merge(df[key+['r']], how='left', on=key)
        dfin.index = dfu.index
        return(dfin.r)
