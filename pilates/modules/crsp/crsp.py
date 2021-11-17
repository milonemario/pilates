"""
Provides processing functions for CRSP data.
"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import numba

from sklearn.linear_model import LinearRegression


class crsp(wrds_module):

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_id = 'permno'
        self.col_date = 'date'
        self.key = [self.col_id, self.col_date]
        # For link with COMPUSTAT
        self.linktype = ['LU', 'LC', 'LS']
        # self.linkprim = ['P', 'C']
        # Default data frequency
        self.freq = 'M'

    def set_frequency(self, frequency):
        if frequency in ['Monthly', 'monthly', 'M', 'm']:
            self.freq = 'M'
            self.sf = self.msf
            self.si = self.msi
        elif frequency in ['Daily', 'daily', 'D', 'd']:
            self.freq = 'D'
            self.sf = self.dsf
            self.si = self.dsi
        else:
            raise Exception('CRSP data frequency should by either',
                            'Monthly or Daily')

    def permno_from_gvkey(self, data):
        """ Returns CRSP permno from COMPUSTAT gvkey.

        This code is insired from WRDS sample program 'merge_funda_crsp_byccm.sas'
        available on the WRDS website.

        Arguments:
            data -- User provided data.
                    Required columns: [gvkey, datadate]
            link_table --   WRDS provided linktable (ccmxpf_lnkhist)
            linktype -- Default: [LC, LU]
            linkprim -- Default: [P, C]
        """
        # Columns required from data
        key = ['gvkey', 'datadate']
        # Columns required from the link table
        cols_link = ['gvkey', 'lpermno', 'linktype', 'linkprim',
                     'linkdt', 'linkenddt']
        # Open the user data
        df = self.open_data(data, key).drop_duplicates().dropna()
        ## Create begin and edn of fiscal year variables
        #df['endfyr'] = df.datadate
        #df['beginfyr'] = (df.datadate - np.timedelta64(11, 'M')).astype('datetime64[M]')
        # Open the link data
        link = self.open_data(self.linktable, cols_link)
        link = link.dropna(subset=['gvkey', 'lpermno', 'linktype', 'linkprim'])
        # Retrieve the specified links
        link = link[(link.linktype.isin(self.linktype))]
        #link = link[['gvkey', 'lpermno', 'linkdt', 'linkenddt']]
        # Merge the data
        dm = df.merge(link, how='left', on='gvkey')
        # Filter the dates (keep correct matches)
        ## Note: Use conditions from WRDS code.
        cond1 = (dm.linkdt <= dm.datadate) | (pd.isna(dm.linkdt))
        cond2 = (dm.datadate <= dm.linkenddt) | (pd.isna(dm.linkenddt))
        dm = dm[cond1 & cond2]

        # Deal with duplicates
        dups = dm[key].duplicated(keep=False)
        dmf = dm[~dups]     # Final links list
        dmd = dm[dups].set_index(['gvkey', 'datadate'])
        ## Favor linkprim, in order: 'P', 'C', 'J' and 'N'
        for lp in ['P', 'C', 'J']:
            dups_lp = dmd[dmd.linkprim==lp]
            dmd = dmd[~dmd.index.isin(dups_lp.index)]
            dups_lp = dups_lp.reset_index()
            dmf = pd.concat([dmf, dups_lp])

        # Rename lpermno to permno and remove unnecessary columns
        dmf = dmf.rename(columns={'lpermno': 'permno'})
        dmf = dmf[['gvkey', 'datadate', 'permno']]
        # Check for duplicates on the key
        n_dup = dmf.shape[0] - dmf[key].drop_duplicates().shape[0]
        if n_dup > 0:
            print("Warning: The merged permno",
                  "contains {:} duplicates".format(n_dup))
        # Add the permno to the user's data
        dfu = self.open_data(data, key).dropna()
        dfin = dfu.merge(dmf, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.permno)

    def permno_from_cusip(self, data):
        """ Returns CRSP permno from CUSIP.

        Note: this function does not ensure a 1-to-1 mapping and there might
        be more than one cusip for a given permno (several cusips may have the
        same permno).

        Args:
            data -- User provided data.
                    Required columns: ['cusip']
                    The cusip needs to be the CRSP ncusip.
        """
        dfu = self.open_data(data, ['cusip'])
        cs = dfu.drop_duplicates()
        pc = self.open_data(self.msenames, ['ncusip', 'permno'])
        pc = pc.drop_duplicates()
        # Merge the permno
        csf = cs.merge(pc, how='left', left_on=['cusip'], right_on=['ncusip'])
        csf = csf[['cusip', 'permno']].dropna().drop_duplicates()
        dfin = dfu.merge(csf, how='left', on='cusip')
        dfin.index = dfu.index
        return(dfin.permno)

    def _adjust_shares(self, data, col_shares):
        """ Adjust the number of shares using CRSP cfacshr field.
        Arguments:
            data -- User provided data.
                    Required fields: [permno, 'col_shares', 'col_date']
            col_shares --   The field with the number of shares from data.
            col_date -- The date field from data to use to compute the
                        adjustment.
        """
        # Open and prepare the user data
        cols = ['permno', col_shares, self.d.col_date]
        dfu = self.open_data(data, cols)
        index = dfu.index
        dt = pd.to_datetime(dfu[self.d.col_date]).dt
        dfu['year'] = dt.year
        dfu['month'] = dt.month
        # Open and prepare the CRSP data
        cols = ['permno', 'date', 'cfacshr']
        df = self.open_data(self.msf, cols)
        dt = pd.to_datetime(df.date).dt
        df['year'] = dt.year
        df['month'] = dt.month
        # Merge the data
        key = ['permno', 'year', 'month']
        dfu = dfu[key+[col_shares]].merge(df[key+['cfacshr']],
                                          how='left', on=key)
        dfu.loc[dfu.cfacshr.isna(), 'cfacshr'] = 1
        # Compute the adjusted shares
        dfu['adj_shares'] = dfu[col_shares] * dfu.cfacshr
        dfu.index = index
        return(dfu.adj_shares.astype('float32'))

    def _get_fields(self, fields, data=None, file=None):
        """ Returns the fields from CRSP.
        This function is only used internally for the CRSP module.
        Arguments:
            fields -- Fields from file_fund
            data -- User provided data
                    Required columns: [permno, date]
                    If none given, returns the entire compustat with key.
                    Otherwise, return only the fields with the data index.
            file -- File to use. Default to stock files
        """
        # Get the fields
        # Note: CRSP data is clean without duplicates
        if not file:
            file = self.sf
        key = [self.col_id, self.col_date]
        if file == self.si:
            key = [self.col_date]
        df = self.open_data(file, key+fields)
        # Construct the object to return
        if data is not None:
            # Merge and return the fields
            data_key = self.open_data(data, key)
            index = data_key.index
            dfin = data_key.merge(df, how='left', on=key)
            dfin.index = index
            return(dfin[fields])
        else:
            # Return the entire dataset with keys
            return(df)

    def get_fields_daily(self, fields, data):
        """ Returns the fields from CRSP daily.
        Arguments:
            fields --   Fields from file_fund
            data -- User provided data
                    Required columns: [permno, date]
                    If none given, returns the entire compustat with key.
                    Otherwise, return only the fields with the data index.
        Requires:
            self.d.col_date -- Date field to use for the user data
        """
        keyu = ['permno', self.d.col_date]
        dfu = self.open_data(data, keyu)
        dfu.loc[:, self.col_date] = dfu[self.d.col_date]
        dfu[fields] = self._get_fields(fields, dfu, self.dsf)
        return(dfu[fields])

    # def _get_window_sort(self, nperiods, caldays, min_periods):
    #     # Define the window and how the data should be sorted
    #     if caldays is None:
    #         window = abs(nperiods)
    #         ascending = (nperiods < 0)
    #     else:
    #         window = str(abs(caldays)) + "D"
    #         ascending = (caldays < 0)
    #         if min_periods is None:
    #             print("Warning: It is advised to provide a minimum number of observations "
    #                   "to compute aggregate values when using the 'caldays' arguments. "
    #                   "No doing so will result in small rolling windows.")
    #     return window, ascending

    def _value_for_data(self, var, data, ascending, useall):
        """" Add values to the users data and return the values.
        Arguments:
            df --   Internal data containing the values
                    Columns: [permno, date]
                    The data is indexed by date and grouped by permno.
            data -- User data
                    Columns: [permno, wrds.col_date]
            nperiods -- Number of periods to compute the variable
            useall --  If True, use the compounded return of the last
                        available trading date (if nperiods<0) or the
                        compounded return of the next available trading day
                        (if nperiods>0).
        """
        key = self.key
        var.name = 'var'
        values = var.reset_index()
        #if nperiods == 0:
        #    values = var.reset_index()
        #else:
        #    # Check that the shift onl occurs within permnos
        #    import ipdb; ipd.set_trace();
        #    values = var.shift(-nperiods).reset_index()
        # Make sure the types are correct
        values = self._correct_columns_types(values)
        # Open user data
        cols_data = [self.col_id, self.d.col_date]
        dfu = self.open_data(data, cols_data)
        # Prepare the dataframes for merging
        dfu = dfu.sort_values(self.d.col_date)
        dfu = dfu.dropna()
        values = values.sort_values(self.col_date)
        if useall:
            # Merge on permno and on closest date
            # Use the last or next trading day if requested
            # Shift a maximum of 6 days
            if ascending:
                direction = 'backward'
            else:
                direction = 'forward'
            dfin = pd.merge_asof(dfu, values,
                                 left_on=self.d.col_date,
                                 right_on=self.col_date,
                                 by=self.col_id,
                                 tolerance=pd.Timedelta('6 day'),
                                 direction=direction)
        else:
            dfin = dfu.merge(values, how='left', left_on=cols_data, right_on=self.key)
        dfin.index = dfu.index
        return(dfin['var'].astype('float32'))

    def _value_for_data_index(self, var, data, ascending, useall):
        """" Add indexes values to the users data and return the values.
        Arguments:
            df --   Internal data containing the values
                    Columns: [permno, date]
                    The data is indexed by date and grouped by permno.
            data -- User data
                    Columns: [permno, wrds.col_date]
            nperiods -- Number of periods to compute the variable
            useall --  If True, use the compounded return of the last
                        available trading date (if nperiods<0) or the
                        compounded return of the next available trading day
                        (if nperiods>0).
        """
        var.name = 'var'
        values = var.reset_index()
        values = self._correct_columns_types(values)
        # Open user data
        cols_data = [self.d.col_date]
        dfu = self.open_data(data, cols_data)
        # Prepare the dataframes for merging
        dfu = dfu.sort_values(self.d.col_date)
        dfu = dfu.dropna()
        values = values.sort_values(self.col_date)
        if useall:
            # Merge on permno and on closest date
            # Use the last or next trading day if requested
            # Shift a maximum of 6 days
            if ascending:
                direction = 'backward'
            else:
                direction = 'forward'
            dfin = pd.merge_asof(dfu, values,
                                 left_on=self.d.col_date,
                                 right_on=self.col_date,
                                 tolerance=pd.Timedelta('6 day'),
                                 direction=direction)
        else:
            dfin = dfu.merge(values, how='left', left_on=cols_data, right_on=self.col_date)
        dfin.index = dfu.index
        return(dfin['var'].astype('float32'))

    ##########################
    # Variables Computations #
    ##########################

    def _tso(self, data):
        """ Compute total share outstanding. """
        cols = ['permno', 'date', 'shrout', 'cfacshr']
        df = self.open_data(self.sf, cols)
        if self.freq == 'M':
            dt = pd.to_datetime(df.date).dt
            df['year'] = dt.year
            df['month'] = dt.month
            key = ['permno', 'year', 'month']
        else:
            key = ['permno', 'date']
        df['tso'] = df.shrout * df.cfacshr * 1000
        dfu = self.open_data(data, ['permno', self.d.col_date])
        if self.freq == 'M':
            dt = pd.to_datetime(dfu[self.d.col_date]).dt
            dfu['year'] = dt.year
            dfu['month'] = dt.month
        else:
            dfu['date'] = dfu[self.d.col_date]
        dfin = dfu.merge(df[key+['tso']], how='left', on=key)
        dfin.index = dfu.index
        return(dfin.tso)

    def compounded_return(self, data, nperiods=1, caldays=None, min_periods=None,
                          logreturn=False, useall=True):
        r"""
        Return the compounded daily returns over 'nperiods' periods.
        If using daily frequency, one period refers to one day.
        If using monthly frequency, one period refers to on month.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            nperiods --    Number of periods to use to compute the compounded
                        returns. If positive, compute the return over
                        'nperiods' in the future. If negative, compute the
                        return over abs(nperiods) in the past.
            useall --  If True, use the compounded return of the last
                        available trading date (if nperiods<0) or the
                        compounded return of the next available trading day
                        (if nperiods>0).
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Check arguments
        if nperiods==0:
            raise Exception("nperiods must be different from 0.")
        # Open the necessary data
        key = self.key
        fields = ['ret']
        sf = self._get_fields(fields)
        # Create the time series index
        sf = sf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the compounded returns
        sf['ln1ret'] = np.log(1 + sf.ret)
        sumln = sf.groupby(self.col_id).rolling(window, min_periods=min_periods).ln1ret.sum()
        if logreturn:
            cret = sumln
        else:
            cret = np.exp(sumln) - 1
        # Return the variable for the user data
        return(self._value_for_data(cret, data, ascending, useall))

    def volatility_return(self, data, nperiods=1, caldays=None, min_periods=None, useall=True):
        r"""
        Return the daily volatility of returns over 'nperiods' periods.
        If using daily frequency, one period refers to one day.
        If using monthly frequency, one period refers to on month.

        Args:
            data -- User data.
                    Required columns: [permno, 'col_date']
            nperiods -- Number of periods to use to compute the volatility.
                        If positive, compute the volatility over
                        'nperiods' in the future. If negative, compute the
                        volatility over abs(nperiods) in the past.
            useall --  If True, use the volatility of the last
                        available trading date (if nperiods<0) or the
                        volatility of the next available trading day
                        (if nperiods>0).
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Open the necessary data
        key = [self.col_id, self.col_date]
        fields = ['ret']
        sf = self._get_fields(fields)
        # Create the time series index
        sf = sf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the volatility
        vol = sf.groupby(self.col_id).rolling(window, min_periods=min_periods).ret.std()
        return(self._value_for_data(vol, data, ascending, useall))

    def average_bas(self, data, nperiods=1, caldays=None, min_periods=None, useall=True, bas=None):
        r"""
        Return the daily average bid-ask spread over 'nperiods' periods.
        If using daily frequency, one period refers to one day.
        If using monthly frequency, one period refers to on month.

        Args:
            data -- User data.
                    Required columns: [permno, 'col_date']
            nperiods -- Number of periods to use to compute the volatility.
                        If positive, compute the average bid-ask spread over
                        'nperiods' in the future. If negative, compute the
                        average bid-ask spread over abs(nperiods) in the past.
            useall --  If True, use the bid-ask spread of the last
                        available trading date (if ndays<0) or the
                        bid-ask spread of the next available trading day
                        (if ndays>0).
            bas --  Type of bid and ask to use. If None, use the fields
                    'bid' and 'ask' from CRSP. If 'lohi', use the fields
                    'bidlo' and 'askhi' from CRSP.
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Open the necessary data
        key = self.key
        fields = ['bid', 'ask']
        if bas is None:
            fs = ['bid', 'ask']
        elif bas == 'lohi':
            fs = ['bidlo', 'askhi']
        else:
            raise Exception("'bas' argument only accepts None or 'lohi'.")
        dsf = self._get_fields(fs)
        dsf.columns = key + fields
        # Create the time series index
        dsf = dsf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the average bid-ask spread
        # (Ask - Bid) / midpoint
        dsf['spread'] = (dsf.ask - dsf.bid) / ((dsf.ask + dsf.bid)/2.)
        # Bid-ask spread cannot be negative
        dsf['spread'] = dsf['spread'].clip(0, None)
        bas = dsf.groupby('permno').rolling(window, min_periods=min_periods).spread.mean()
        return(self._value_for_data(bas, data, ascending, useall))

    def turnover(self, data, nperiods=1, caldays=None, min_periods=None, useall=True):
        r""" Return the daily turnover over 'nperiods' (usually days).

        Args:
            data (DataFrame):   User data.
                                Required columns: [permno, 'col_date']

            nperiods (int):     Number of periods (usually days)
                                to use to compute the turnover.
                                If positive, compute the turnover over
                                'nperiods' in the future. If negative, compute the
                                turnover over abs(nperiods) in the past.

            useall (bool):      If True, use the turnover of the last
                                available trading date (if nperiods<0) or the
                                turnover of the next available trading day
                                (if nperiods>0).

        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Open the necessary data
        key = self.key
        fields = ['shrout', 'vol']
        # Note: The number of shares outstanding (shrout) is in thousands.
        # Note: The volume in the daily data is expressed in units of shares.
        dsf = self._get_fields(fields)
        # Some type conversion to make sure the rolling window will work.
        dsf['shrout'] = dsf.shrout.astype('float32')
        # Create the time series index
        dsf = dsf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the average turnover
        dsf['vol_sh'] = dsf.vol / (dsf.shrout * 1000)
        turnover = dsf.groupby(self.col_id).rolling(window, min_periods=min_periods).vol_sh.mean()
        return(self._value_for_data(turnover, data, ascending, useall))

    def turnover_shu2000(self, data, nperiods=1, caldays=None, min_periods=None, useall=True):
        r"""
        Return the daily turnover over 'ndays' days.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            col_date -- Column of the dates at which to compute the turnover.
            ndays --    Number of days to use to compute the turnover.
                        If positive, compute the turnover over
                        'ndays' in the future. If negative, compute the
                        turnover over abs(ndays) in the past.
            useall --  If True, use the turnover of the last
                        available trading date (if ndays<0) or the
                        turnover of the next available trading day
                        (if ndays>0).
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Open the necessary data
        key = self.key
        fields = ['shrout', 'vol']
        # Note: The number of shares outstanding (shrout) is in thousands.
        # Note: The volume in the daily data is expressed in units of shares.
        dsf = self._get_fields(fields)
        # Create the time series index
        dsf = dsf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the average turnover
        dsf['vol_sh'] = dsf.vol / (dsf.shrout * 1000)
        dsf['onemvs'] = 1. - dsf.vol_sh
        turnover = 1 - dsf.groupby(self.col_id).rolling(window, min_periods=min_periods).onemvs.apply(np.prod, raw=True)
        return(self._value_for_data(turnover, data, ascending, useall))

    def age(self, data):
        """ Age of the firm - Number of years with return history.

        When no information is available from CRSP, computes the age with
        the user data (assumes the user data is 'complete'). Otherwise, age
        is the max of the user data's age and CRSP age.

        """
        # Use the stocknames table to obtain the date range of permno identifiers
        sn = self.open_data(self.stocknames, [self.col_id, 'namedt', 'nameenddt'])
        # Open the user data
        cols_data = [self.d.col_id, self.d.col_date, self.col_id]
        dfu = self.open_data(data, cols_data)
        index = dfu.index
        # Compte age using user data only first
        dfin = dfu.copy()
        dfin['mindate'] = dfu.groupby(self.d.col_id)[self.d.col_date].transform('min')
        dfin['age_data'] = dfin[self.d.col_date] - dfin.mindate
        # Compute the age with CRSP data
        ## Min date per permno
        sn = sn.groupby(self.col_id)['namedt'].min().reset_index()
        sn = self._correct_columns_types(sn)
        ## Merge start with user data
        dfin = dfin.merge(sn, how='left', on='permno')
        ## Compute age with earliest crsp date by user data id (gvkey if so)
        dfin['age_crsp'] = dfin[self.d.col_date] - dfin.namedt
        # Get the final age (max of data and crsp)
        dfin['age'] = dfin.age_data
        dfin.loc[dfin.age_crsp > dfin.age, 'age'] = dfin.age_crsp
        dfin['age'] = dfin.age.dt.days / 365.
        dfin.index = dfu.index
        return(dfin.age.astype('float32'))

    def beta(self, data, nperiods=1, caldays=None, min_periods=None, useall=True):
        r"""
        Return the beta coefficient computed over 'ndays' days.
        The beta is the slope of the regression of daily stock returns
        on equal-weighted market returns.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            col_date -- Column of the dates at which to compute the value.
            ndays --    Number of days to use to compute the value.
                        If positive, compute over 'ndays' in the future.
                        If negative, compute over abs(ndays) in the past.
            useall --  If True, use the value of the last
                        available trading date (if ndays<0) or the
                        value of the next available trading day
                        (if ndays>0).
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Open the necessary data
        key = self.key
        ret = 'ret'
        mret = 'vwretd'
        dsf = self._get_fields([ret])
        # Get the value-weighted market returns
        dsf[mret] = self._get_fields([mret], data=dsf, file=self.dsi)
        # Create the time series index
        dsf = dsf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the rolling beta

        def compute_beta(dfg):
            cov = dfg[[ret, mret]].rolling(window=window, min_periods=min_periods).cov()
            cov = cov.unstack()[mret] # Gives cov(ret, mret) and cov(mret, mret)
            beta = cov[ret] / cov[mret]
            return beta

        beta = dsf.groupby(self.col_id).apply(compute_beta)
        return(self._value_for_data(beta, data, ascending, useall))

    def delist(self, data, caldays=1):
        r"""
        Return the delist dummy.
        Delist is at one if the firm is delisted within the next 'caldays' days
        for financial difficulties.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            col_date -- Column of the dates at which to compute the value.
            caldays --  Number of calendar days to consider.
                        The delist dummy equals one if there is a delisting in
                        the following caldays calendar days.
        """
        # Open the necessary data
        key = self.key
        dse = self._get_fields(['dlstcd'], file=self.dse)
        # Keep the correct delisting codes (for financial difficulties)
        codes = [400, 401, 403, 450, 460, 470, 480, 490,
                 552, 560, 561, 572, 574, 580, 582]
        # Remove missing values
        dse = dse.dropna()
        # Convert field to Integer
        # dse.dlstcd = dse.dlstcd.astype(int)
        # Create the delist dummy
        dse.loc[dse.dlstcd.isin(codes), 'delist'] = 1
        dse = dse[dse.delist==1]
        dse['startdate'] = dse.date - pd.to_timedelta(caldays, unit='d')
        dse['enddate'] = dse.date
        dse = dse[['permno','startdate', 'enddate']]
        # Open user data
        cols_data = [self.col_id, self.d.col_date]
        dfu = self.open_data(data, cols_data)
        index = dfu.index
        # Merge the dates and create the delist dummy
        dfin = dfu.merge(dse, how='left', on=self.col_id)
        dfin['delist'] = 0
        dfin.loc[(dfin[self.d.col_date]>=dfin.startdate) &
                 (dfin[self.d.col_date]<=dfin.enddate), 'delist'] = 1
        dfin.index = index
        return(dfin.delist)

    def crashrisk_hmt(self, data):
        """ Crash Risk as defined in Hutton, Marcus and Tehranian (JFE 2009).

        Note: Computing the crash risk requires both CRSP and Compustat data
        as one needs to identify fiscal years. Therefore the availability of
        this variable depends on the match CRSP-Compustat.

        It is therefore recommended to add the result of this funtion to
        Compustat data.

        Arguments:
            data -- User data.
                    Required columns: [permno, datadate]

        """

        # Open the necessary data
        ## Get the returns
        df = self._get_fields(['ret'])
        ## Get the value-weighted market index
        df['vwretd'] = self._get_fields(['vwretd'], data=df, file=self.dsi)

        # Identify the weeks
        df['year'] = df.date.dt.year
        df['week'] = df.date.dt.isocalendar().week
        df['yw'] = df.year*100 + df.week
        df.drop(columns=['year', 'week'], inplace=True)

        # Compute weekly returns for stock and index
        df['ln1ret'] = np.log(1+df.ret)
        df['ln1vwret'] = np.log(1+df.vwretd)
        ln1ret_week = df.groupby([self.col_id, 'yw']).ln1ret.sum()
        ln1vwret_week = df.groupby([self.col_id, 'yw']).ln1vwret.sum()
        ret_week = np.exp(ln1ret_week) - 1
        vwret_week = np.exp(ln1vwret_week) - 1
        dfret = pd.concat([ret_week, vwret_week], axis=1)
        dfret.columns = ['ret', 'vwret']

        # Regress stock returns on index (with lead and lag terms)
        dfret['vwret_lag1'] = dfret.groupby(level='permno')['vwret'].shift(1)
        dfret['vwret_lead1'] = dfret.groupby(level='permno')['vwret'].shift(-1)
        dfret = dfret.reset_index()
        # Drop missing values
        dfret = dfret.dropna()

        # Compute firm-specific weekly returns
        ## Note: one regression by permno

        def spec_week_ret(dfg):
            y = dfg.ret
            X = dfg[['vwret_lag1', 'vwret', 'vwret_lead1']]
            m = LinearRegression().fit(X, y)
            return np.log(1 + (y-m.predict(X)))

        dfsret = dfret.groupby('permno').apply(spec_week_ret)
        dfsret = dfsret.reset_index()
        dfsret.index = dfsret.level_1
        # Add the specific weekly return to the return data
        dfret['specret'] = dfsret['ret']

        ## Get compustat fiscal years
        comp_freq = self.d.comp.freq
        self.d.comp.set_frequency('Annual')
        df_comp = self.d.comp.get_fields([])
        self.d.comp.set_frequency(comp_freq)
        ## Add permno
        df_comp['permno'] = self.permno_from_gvkey(df_comp)
        #df_comp['datadate_m'] = df_comp.datadate
        df_comp = df_comp[['permno', 'datadate']]
        ## Merge compustat datadate to CRSP data
        ### Create a date column corresponding to the end of the year-week.
        dfret['year'] = (dfret.yw / 100).astype(int)
        dfret['week'] = dfret.yw - (dfret.year*100)
        dfret['ydt'] = pd.to_datetime(dfret.year, format='%Y')
        dfret['date'] = dfret.ydt + pd.to_timedelta(dfret.week, unit='W')

        # Create the final return data to merge with Compustat
        dfretm = dfret[['permno', 'date', 'specret']]

        ### Match all CRSP date within a year before datadate as belonging
        ### to that fiscal year.
        dfretm = dfretm.sort_values('date')
        df_comp = df_comp.dropna()
        df_comp = df_comp.sort_values('datadate')
        dfretm = self._correct_columns_types(dfretm)
        dfm = pd.merge_asof(dfretm, df_comp,
                            left_on='date',
                            right_on='datadate',
                            by='permno',
                            tolerance=pd.Timedelta('364 day'),
                            direction='forward')
        ### Keep observations with datadate
        dfm = dfm[dfm.datadate.notna()]

        # Compute crash-risk dummy
        ## Compute mean and stdev of specific firm return over the fiscal year
        dfm['specret_mean'] = dfm.groupby(['permno', 'datadate'])['specret'].transform('mean')
        dfm['specret_sd'] = dfm.groupby(['permno', 'datadate'])['specret'].transform('std')
        dfm['crash'] = 0
        cond = (dfm.specret < dfm.specret_mean - 3.09*dfm.specret_sd)
        dfm.loc[cond, 'crash'] = 1

        # Merge to user data
        key = ['permno', 'datadate']
        dfu = self.open_data(data, key)
        index = dfu.index
        # Final crash dummy
        df_crash = dfm.groupby(key)['crash'].max().reset_index()
        df_crash = self._correct_columns_types(df_crash)
        # Merge the dates and create the delist dummy
        dfin = dfu.merge(df_crash, how='left', on=key)
        dfin.index = index
        return(dfin.crash.astype('float32'))

    def ncskew_chs(self, data, nperiods=1, caldays=None, min_periods=None, useall=True):
        """ Negative Coefficient of Skewness as defined in Chen, Hong and Stein (JFE 2001).

        Arguments:
            data        User data.
                        Required columns: [permno, 'col_date']
            nperiods    Number of periods to compute the measure
                        (usually in trading days).

        """

        # Open the necessary data
        key = self.key
        fields = ['ret']
        df = self._get_fields(fields)
        df['vwretd'] = self._get_fields(['vwretd'], data=df, file=self.dsi)

        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Create the time series index in the correct order
        df = df.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute market adjusted returns (log changes in price)
        df['lnret'] = np.log(1+df.ret) - np.log(1+df.vwretd)

        # Function to compute ncskew given a series of returns
        @numba.jit(nopython=True)
        def compute_ncskew(lnret):
            # Demean returns
            ret_dem = lnret - np.nanmean(lnret)
            # Compute sums of squares and cubes
            sum_ret2 = np.nansum(ret_dem**2)
            sum_ret3 = np.nansum(ret_dem**3)
            # Compute NCSKEW
            n = lnret.size - np.count_nonzero(np.isnan(lnret))
            ncskew = -( n * ((n-1)**(3./2.)) * sum_ret3 ) / ( (n-1) * (n-2) * (sum_ret2**(3./2.)) )
            return ncskew

        ncskew = df.groupby(self.col_id).rolling(window, min_periods=min_periods).lnret.apply(compute_ncskew, raw=True)

        # Return the variable for the user data
        return(self._value_for_data(ncskew, data, ascending, useall))

    def duvol_chs(self, data, nperiods=1, caldays=None, min_periods=None, useall=True):
        """ Down-to-Up Volatility as defined in Chen, Hong and Stein (JFE 2001).

        Arguments:
            data        User data.
                        Required columns: [permno, 'col_date']

        """

        # Open the necessary data
        key = self.key
        fields = ['ret']
        df = self._get_fields(fields)
        df['vwretd'] = self._get_fields(['vwretd'], data=df, file=self.dsi)

        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Create the time series index in the correct order
        df = df.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute market adjusted returns (log changes in price)
        df['lnret'] = np.log(1+df.ret) - np.log(1+df.vwretd)

        # Function to compute ncskew given a series of returns
        @numba.jit(nopython=True)
        def compute_duvol(lnret):
            # Compute the mean and split Ups and Downs
            mean = np.nanmean(lnret)
            ups = lnret[lnret >= mean]
            downs = lnret[lnret < mean]
            # Compute sum of squares
            su2 = np.nansum(ups**2)
            sd2 = np.nansum(downs**2)
            # Compute DUVOL
            nu = len(ups)
            nd = len(downs)
            duvol = np.log( ((nu-1)*su2) / ((nd-1)*sd2) )
            return duvol

        ncskew = df.groupby(self.col_id).rolling(window, min_periods=min_periods).lnret.apply(compute_duvol, raw=True)

        # Return the variable for the user data
        return(self._value_for_data(ncskew, data, ascending, useall))

    ###########
    # Indexes #
    ###########

    def compounded_return_index(self, data, nperiods=1, caldays=None, min_periods=None, useall=True, field='vwretd'):
        r"""
        Return the compounded daily returns of CRSP index over 'nperiods' periods.
        If using daily frequency, one period refers to one day.
        If using monthly frequency, one period refers to on month.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            nperiods --    Number of periods to use to compute the compounded
                        returns. If positive, compute the return over
                        'nperiods' in the future. If negative, compute the
                        return over abs(nperiods) in the past.
            useall --  If True, use the compounded return of the last
                        available trading date (if nperiods<0) or the
                        compounded return of the next available trading day
                        (if nperiods>0).
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Check arguments
        if nperiods==0:
            raise Exception("nperiods must be different from 0.")
        # Open the necessary data
        key = self.key
        fields = [field]
        sf = self._get_fields(fields, file=self.si)
        # Create the time series index
        sf = sf.set_index(self.col_date).sort_index(ascending=ascending)
        # Compute the compounded returns
        sf['ln1ret'] = np.log(1 + sf[field])
        sumln = sf.rolling(window, min_periods=min_periods).ln1ret.sum()
        cret = np.exp(sumln) - 1
        # Return the variable for the user data
        return(self._value_for_data_index(cret, data, ascending, useall))
