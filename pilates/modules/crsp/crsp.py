"""
Provides processing functions for CRSP data.
"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal


class crsp(wrds_module):

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_id = 'permno'
        self.col_date = 'date'
        self.key = [self.col_id, self.col_date]
        # For link with COMPUSTAT
        self.linktype = ['LC', 'LU']
        self.linkprim = ['P', 'C']
        # Default data frequency
        self.freq = 'M'

    def set_frequency(self, frequency):
        if frequency in ['Monthly', 'monthly', 'M', 'm']:
            self.freq = 'M'
            self.sf = self.msf
        elif frequency in ['Daily', 'daily', 'D', 'd']:
            self.freq = 'D'
            self.sf = self.dsf
        else:
            raise Exception('CRSP data frequency should by either',
                            'Monthly or Daily')

    def permno_from_gvkey(self, data):
        """ Returns CRSP permno from COMPUSTAT gvkey.
        Arguments:
            data -- User provided data.
                    Required columns: [gvkey, 'col_date']
            link_table --   WRDS provided linktable (ccmxpf_lnkhist)
            linktype -- Default: [LC, LU]
            linkprim -- Default: [P, C]
        """
        # Columns required from data
        key = ['gvkey', self.d.col_date]
        # Columns required from the link table
        cols_link = ['gvkey', 'lpermno', 'linktype', 'linkprim',
                     'linkdt', 'linkenddt']
        # Open the data
        data_key = self.open_data(data, key).drop_duplicates().dropna()
        link = self.open_data(self.linktable, cols_link)
        link = link.dropna(subset=['gvkey', 'lpermno', 'linktype', 'linkprim'])
        # Retrieve the specified links
        link = link[(link.linktype.isin(self.linktype)) &
                    (link.linkprim.isin(self.linkprim))]
        link = link[['gvkey', 'lpermno', 'linkdt', 'linkenddt']]
        # Merge the data
        dm = data_key.merge(link, how='left', on='gvkey')
        # Filter the dates (keep correct matches)
        cond1 = (pd.notna(dm.linkenddt) &
                 (dm[self.d.col_date] >= dm.linkdt) &
                 (dm[self.d.col_date] <= dm.linkenddt))
        cond2 = (pd.isna(dm.linkenddt) & (dm['datadate'] >= dm.linkdt))
        dm = dm[cond1 | cond2]
        # Rename lpermno to permno and remove unnecessary columns
        dm = dm.rename(columns={'lpermno': 'permno'})
        dm = dm[['gvkey', self.d.col_date, 'permno']]
        # Check for duplicates on the key
        n_dup = dm.shape[0] - dm[key].drop_duplicates().shape[0]
        if n_dup > 0:
            print("Warning: The merged permno",
                  "contains {:} duplicates".format(n_dup))
        # Add the permno to the user's data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(dm, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.permno.astype('float32'))

    def permno_from_cusip(self, data):
        """ Returns CRSP permno from CUSIP.
        Arguments:
            data -- User provided data.
                    Required columns: ['cusip']
                    The cusip needs to be the CRSP ncusip.
        """
        dfu = self.d.open_data(data, ['cusip'])
        cs = dfu.drop_duplicates()
        pc = self.d.open_data(self.msenames, ['ncusip', 'permno'])
        pc = pc.drop_duplicates()
        cs = cs.merge(pc, how='left', left_on=['cusip'], right_on=['ncusip'])
        csf = cs[['cusip', 'permno']].dropna().drop_duplicates()
        dfin = dfu.merge(csf, how='left', on='cusip')
        dfin.index = dfu.index
        return(dfin.permno.astype('float32'))

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
        if file is None:
            file = self.sf
        key = [self.col_id, self.col_date]
        if file == self.dsi:
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

    def _closest_trading_date(self, dates, t='past'):
        """ Return the closest trading day either in the past (t='past')
        or in the future (t='future').
        Based on opening days of the NYSE. """
        dates = pd.to_datetime(dates)
        nyse = mcal.get_calendar('NYSE')
        mind = dates.min()
        maxd = dates.max()
        dates_nyse = nyse.schedule(mind, maxd).market_open.dt.date
        sm = (~dates.dt.date.isin(dates_nyse)).astype(int)
        for i in range(0, 6):
            if t == 'past':
                dates = dates - pd.to_timedelta(sm, unit='d')
            else:
                dates = dates + pd.to_timedelta(sm, unit='d')
            sm = (~dates.dt.date.isin(dates_nyse)).astype(int)
        return(dates.dt.date)

    def _value_for_data(self, var, data, nperiods, useall):
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
        if nperiods < 0:
            values = var.reset_index()
        else:
            values = var.shift(-nperiods).reset_index()
        # Open user data
        cols_data = [self.col_id, self.d.col_date]
        dfu = self.open_data(data, cols_data)
        index = dfu.index
        # Use the last or next trading day if requested
        # Shift a maximum of 6 days
        if useall is True:
            t = 'past'
            if nperiods > 0:
                t = 'future'
            dfu[self.col_date] = self._closest_trading_date(dfu[self.d.col_date], t=t)
        else:
            dfu[self.col_date] = dfu[self.d.col_date]
        # Merge the cumulative return to the data
        dfin = dfu.merge(values, how='left', on=key)
        dfin.index = index
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

    def compounded_return(self, data, nperiods=1, useall=True):
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
        # Check arguments
        if nperiods==0:
            raise Exception("nperiods must be different from 0.")
        # Open the necessary data
        key = self.key
        fields = ['ret']
        sf = self._get_fields(fields)
        # Create the time series index
        sf = sf.set_index(self.col_date)
        # Compute the compounded returns
        sf['ln1ret'] = np.log(1 + sf.ret)
        window = abs(nperiods)
        sumln = sf.groupby(self.col_id).rolling(window).ln1ret.sum()
        cret = np.exp(sumln) - 1
        # Reeturn the variable for the user data
        return(self._value_for_data(cret, data, nperiods, useall))

    def volatility_return(self, data, nperiods=1, useall=True):
        r"""
        Return the daily volatility of returns over 'nperiods' periods.
        If using daily frequency, one period refers to one day.
        If using monthly frequency, one period refers to on month.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            col_date -- Column of the dates at which to compute the return
            nperiods -- Number of periods to use to compute the volatility.
                        If positive, compute the volatility over
                        'nperiods' in the future. If negative, compute the
                        volatility over abs(nperiods) in the past.
            useall --  If True, use the volatility of the last
                        available trading date (if nperiods<0) or the
                        volatility of the next available trading day
                        (if nperiods>0).
        """
        # Open the necessary data
        key = [self.col_id, self.col_date]
        fields = ['ret']
        sf = self._get_fields(fields)
        # Create the time series index
        sf = sf.set_index(self.col_date)
        # Compute the volatility
        window = abs(nperiods)
        vol = sf.groupby(self.col_id).rolling(window).ret.std()
        return(self._value_for_data(vol, data, nperiods, useall))

    def average_bas(self, data, nperiods=1, useall=True, bas=None):
        r"""
        Return the daily average bid-ask spread over 'ndays' days.
        Arguments:
            data -- User data.
                    Required columns: [permno, 'col_date']
            col_date -- Column of the dates at which to compute the return
            ndays --    Number of days to use to compute the bid-ask spread.
                        If positive, compute the bid-ask spread over
                        'ndays' in the future. If negative, compute the
                        bid-ask spread over abs(ndays) in the past.
            useall --  If True, use the bid-ask spread of the last
                        available trading date (if ndays<0) or the
                        bid-ask spread of the next available trading day
                        (if ndays>0).
            bas --  Type of bid and ask to use. If None, use the fields
                    'bid' and 'ask' from CRSP. If 'lohi', use the fields
                    'bidlo' and 'askhi' from CRSP.
        """
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
        dsf = dsf.set_index(self.col_date)
        # Compute the average bid-ask spread
        # (Ask - Bid) / midpoint
        dsf['spread'] = (dsf.ask - dsf.bid) / ((dsf.ask + dsf.bid)/2.)
        window = abs(nperiods)
        bas = dsf.groupby('permno').rolling(window).spread.mean()
        return(self._value_for_data(bas, data, nperiods, useall))

    def turnover(self, data, nperiods=1, useall=True):
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
        # Open the necessary data
        key = self.key
        fields = ['shrout', 'vol']
        # Note: The number of shares outstanding (shrout) is in thousands.
        # Note: The volume in the daily data is expressed in units of shares.
        dsf = self._get_fields(fields)
        # Create the time series index
        dsf = dsf.set_index('date')
        # Compute the average turnover
        dsf['vol_sh'] = dsf.vol / (dsf.shrout * 1000)
        window = abs(nperiods)
        turnover = dsf.groupby('permno').rolling(window).vol_sh.mean()
        return(self._value_for_data(turnover, data, nperiods, useall))

    def turnover_shu2000(self, data, nperiods=1, useall=True):
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
        # Open the necessary data
        key = self.key
        fields = ['shrout', 'vol']
        # Note: The number of shares outstanding (shrout) is in thousands.
        # Note: The volume in the daily data is expressed in units of shares.
        dsf = self._get_fields(fields)
        # Create the time series index
        dsf = dsf.set_index('date')
        # Compute the average turnover
        dsf['vol_sh'] = dsf.vol / (dsf.shrout * 1000)
        dsf['onemvs'] = 1. - dsf.vol_sh
        window = abs(nperiods)
        turnover = 1 - dsf.groupby('permno').rolling(window).onemvs.apply(np.prod, raw=True)
        return(self._value_for_data(turnover, data, nperiods, useall))

    def age(self, data):
        """ Age of the firm - Number of years with return history. """
        # Open the necessary data
        key = [self.col_id, self.col_date]
        fields = ['ret']
        sf = self._get_fields(fields)
        mindate = sf.groupby(self.col_id).date.min()
        mindate.name = 'mindate'
        # Merge the min date to user data
        cols_data = [self.col_id, self.d.col_date]
        dfu = self.open_data(data, cols_data)
        index = dfu.index
        dfin = dfu.merge(mindate, how='left', on=self.col_id)
        dfin['age'] = (dfin[self.d.col_date] - dfin.mindate).dt.days / 365.
        dfin.index = index
        return(dfin.age)

    def beta(self, data, ndays=1, useall=True):
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
        # Open the necessary data
        key = self.key
        ret = 'ret'
        mret = 'vwretd'
        dsf = self._get_fields([ret])
        # Get the value-weighted market returns
        dsf[mret] = self._get_fields([mret], data=dsf, file=self.dsi)
        # Create the time series index
        dsf = dsf.set_index([self.col_date])
        # Compute the rolling beta
        #import ipdb; ipdb.set_trace();
        window = abs(ndays)
        cov = dsf.groupby(self.col_id).rolling(window)[[ret,mret]].cov()
        cov = cov.unstack()[mret] # Gives cov(ret, mret) and cov(mret, mret)
        beta = cov[ret] / cov[mret]
        return(self._value_for_data(beta, data, ndays, useall))

    def delist(self, data, caldays=1):
        r"""
        Return the delist dummy.
        Delist is at one if the firm is delisted within the next 'ndays' days
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
        dse.dlstcd = dse.dlstcd.astype(int)
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
