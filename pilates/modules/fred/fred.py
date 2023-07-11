"""
Provide convenient functions to get FRED economic data.

Series can be found here:
https://fred.stlouisfed.org/
"""

from pilates import data_module
import pandas as pd
import numpy as np
from fredapi import Fred

FRED_API_KEY = '69e42ccbb7fa5da6cc743e564d08be62'

class fred(data_module):

    def __init__(self, w):
        data_module.__init__(self, w)

    #######################################
    # Make available some fredapi methods #
    #######################################

    def get_series_info(self, series):
        self.fred = Fred(api_key = FRED_API_KEY)
        return self.fred.get_series_info(series)

    ##########################################
    # Additional generic methods for pilates #
    ##########################################

    def get_series_for_data(self, data, series, col_date=None, tolerance=None):
        if col_date is None:
            # Use the date set at the library level
            col_date = self.d.col_date
        if tolerance is None:
            # Series is given as a name
            fred_series = self.fred.get_series(series)
            # Get series information (frequency, etc)
            df_fred_info = self.fred.get_series_info(series)
            freq = df_fred_info['frequency_short']
            if freq == 'A':
                tolerance = pd.Timedelta('370 day')
            elif freq == 'Q':
                tolerance = pd.Timedelta('100 day')
            elif freq == 'M':
                tolerance = pd.Timedelta('40 day')
            elif freq == 'D':
                tolerance = pd.Timedelta('2 day')
        else:
            # Series is given as data
            fred_series = series
            series = 'series_name'

        df = pd.DataFrame(fred_series).reset_index()
        df.columns = ['date_fred', series]
        # Prepare the user data
        data[col_date] = pd.to_datetime(data[col_date])
        # Find nearest dates
        data = data.sort_values(col_date)
        df = df.sort_values('date_fred')
        data = data[np.isfinite(data[col_date])]
        dfin = pd.merge_asof(data[col_date], df, left_on=col_date, right_on='date_fred',
                             direction='nearest',
                             tolerance=tolerance)
        dfin.index = data.index
        return dfin[series].astype('float32')

    def get_series(self, series, data=None, col_date=None):
        """ Return the FRED series to be added to the user data.
        """
        # Connection to FRED (connect just before downloading series
        # to avoid Error 504: Gateway Time-out)
        self.fred = Fred(api_key = FRED_API_KEY)
        if data is None:
            return self.fred.get_series(series)
        if col_date is None:
            # Use the date set at the library level
            col_date = self.d.col_date
        dfin = data[[col_date]]
        for s in series:
            dfin[s] = self.get_series_for_data(dfin, s, col_date)
        return dfin[series]


    # Depreciated
    def __get_10y_US_rates(self, data, col_date=None):
        """ Return the 10 years treasury rates.
        Depreciated. Use get_series().
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
