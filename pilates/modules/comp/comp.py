"""
Module to provide processing functions for Compustat data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np


class comp(wrds_module):
    """ Class providing main processing methods for the Compustat data.

    One instance of this classs is automatically created and accessible from
    any data object instance.

    Args:
        w (pilates.data): Instance of pilates.data object.

    Attributes:
        indfmt (list): Specify the industry format to use.
            Correspond to the Compustat field 'indfmt'.
            Defaults to ['INDL']
        datafmt (list): Specify the data format to use.
            Correspond to the Compustat field 'datafmt'.
            Defaults to ['STD']
        consol (list): Specify the consolidation.
            Correspond to the Compustat field 'consol'.
            Defaults to ['C'] (consolidated)
        popsrc (list): Specify the population of firms.
            Correspond to the Compustat field 'popsrc'.
            Defaults to ['D'] (domestic)

    """

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_id = 'gvkey'
        self.col_date = 'datadate'
        self.key = [self.col_id, self.col_date]
        self.freq = None  # Data frequency
        # Filters
        self.indfmt = ['INDL']
        self.datafmt = ['STD']
        self.consol = ['C']
        self.popsrc = ['D']

    def set_date_column(self, col_date):
        self.col_date = col_date
        self.key = [self.col_id, self.col_date]

    def set_frequency(self, frequency):
        """ Define the frequency to use for the Compustat data.
        The corresponding Compustat data 'fund' file (funda or fundq)
        must be added to the module before setting the frequency.

        Args:
            frequency (str): Frequency of the data.
                Supports 'Quarterly' and 'Annual'.

        Raises:
            Exception: If the frequency is not supported.

        """
        if frequency in ['Quarterly', 'quarterly', 'Q', 'q']:
            self.freq = 'Q'
            self.fund = self.fundq
        elif frequency in ['Annual', 'annual', 'A', 'a',
        'Yearly', 'yearly', 'Y', 'y']:
            self.freq = 'A'
            self.fund = self.funda
        else:
            raise Exception('Compustat data frequency should by either',
                            'Annual or Quarterly')

    def _get_fund_fields(self, fields, data, lag=0):
        """ Returns fields from COMPUSTAT 'fund' file.
        The 'fund' file is the Compustat Fundamental data (fundq or funda).

        Args:
            fields (list): List of fields names.
                The fields must exist in the corresponding fund file.
            data (str or pandas.DataFrame): User provided data
                Required columns: [gvkey, datadate]

        Returns:
            pandas.Serie(s): pandas Series to be added to the data.
                The index of the series returned corresponds to the index
                of the provided data.

        """
        key = self.key
        cols_req = ['indfmt', 'datafmt', 'consol', 'popsrc']
        data_key = self.open_data(data, key)
        index = data_key.index
        comp = self.open_data(self.fund,
                                key+fields+cols_req).drop_duplicates()
        # Filter the fund data
        comp = comp[comp.indfmt.isin(self.indfmt) &
                    comp.datafmt.isin(self.datafmt) &
                    comp.consol.isin(self.consol) &
                    comp.popsrc.isin(self.popsrc)]
        comp = comp[key+fields]
        # Remove duplicates
        dup = comp[key].duplicated(keep=False)
        nd0 = comp[~dup]
        dd = comp[dup][key+fields]
        # 1. When duplicates, keep the ones with the most information
        dd['n'] = dd.count(axis=1)
        maxn = dd.groupby(key).n.max().reset_index()
        maxn = maxn.rename({'n': 'maxn'}, axis=1)
        dd = dd.merge(maxn, how='left', on=key)
        dd = dd[dd.n == dd.maxn]
        dup = dd[key].duplicated(keep=False)
        nd1 = dd[~dup][key+fields]
        dd = dd[dup]
        # 2. Choose randomly
        dd['rn'] = dd.groupby(key).cumcount()
        dd = dd[dd.rn == 0]
        nd2 = dd[key+fields]
        # Concatenate the non duplicates
        nd = pd.concat([nd0, nd1, nd2]).sort_values(key)
        # Get the lags if asked for
        ndl = nd.groupby(self.col_id)[fields].shift(lag)
        nd[fields] = ndl
        # Merge and return the fields
        dfin = data_key.merge(nd, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def _get_names_fields(self, fields, data):
        """ Returns fields from COMPUSTAT 'names' file.
        The 'names' file is the Compustat names data.

        Args:
            fields (list): List of fields names.
                The fields must exist in the names file.
            data (str or pandas.DataFrame): User provided data
                Required columns: [gvkey, datadate]

        Returns:
            pandas.Serie(s): pandas Series to be added to the data.
                The index of the series returned corresponds to the index
                of the provided data.

        """
        key = [self.col_id]
        data_key = self.open_data(data, key)
        index = data_key.index
        comp = self.open_data(self.names, key+fields).drop_duplicates()
        # Note: thereare no duplicates in the name file
        # Merge and return the fields
        dfin = data_key.merge(comp, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def _get_adsprate_fields(self, fields, data):
        """ Returns fields from COMPUSTAT 'adsprate' file.
        adsprate file contains S&P debt ratings.
        """
        data_key = self.open_data(data, self.key)
        comp = self.open_data(self.adsprate, self.key+fields).drop_duplicates()
        # Merge and return the fields
        dfin = data_key.merge(comp, how='left', on=self.key)
        dfin.index = data_key.index
        return(dfin[fields])

    def get_fields(self, fields, data=None, lag=0):
        """ Returns fields from COMPUSTAT data.

        Args:
            fields (list): List of fields names.
                The fields may be fields presents wither in the 'fund' file
                or the 'names' file, or may be defined by one of the module
                function.
            data (str or pandas.DataFrame, optional): User provided data
                Required columns: [gvkey, datadate]
            lag (int, optional): If lag = N>0, returns the Nth lag of the
                fields. Can be positive (returns future values).

        Returns:
            pandas.Serie(s) or pandas.DataFrame:
                When data is provided, returns pandas Series to be added to
                the data. The index of the series returned corresponds to the
                index of the provided data.
                When data is not provided, returns a pandas DataFrame with the
                fields and the ['gvkey', 'datadate'] columns.

        """
        key = self.key
        # Determine the raw and additional fields
        cols_fund = self.get_fields_names(self.fund)
        cols_names = self.get_fields_names(self.names)
        cols_adsprate = self.get_fields_names(self.adsprate)
        fields = [f for f in fields if f not in key]
        # fields_toc = [f for f in fields if (f not in fund_raw and
        #                                     f not in names_raw)]
        # Keep the full compustat key (to correctly compute lags)
        df = self.open_data(self.fund, key).drop_duplicates()
        # Get additional fields first (to ensure overwritten fields)
        fields_add = []
        for f in fields:
            if hasattr(self, '_' + f):
                # print(f)
                fn = getattr(self, '_' + f)
                df[f] = fn(df)
                fields_add += [f]
        fund_raw = [f for f in fields if (f in cols_fund and
                                          f not in fields_add)]
        names_raw = [f for f in fields if (f in cols_names and
                                           f not in fields_add)]
        adsprate_raw = [f for f in fields if (f in cols_adsprate and
                                              f not in fields_add)]
        # Get the raw fields for the fund file
        if len(fund_raw) > 0:
            df[fund_raw] = self._get_fund_fields(fund_raw, df)
        # Get the raw fields for the names file
        if len(names_raw) > 0:
            df[names_raw] = self._get_names_fields(names_raw, df)
        # Get the raw fields for the adsprate file
        if len(adsprate_raw) > 0:
            df[adsprate_raw] = self._get_adsprate_fields(adsprate_raw, df)
        # Get the lags if asked for
        if len(fields) > 0 and lag!=0:
            df[fields] = self.get_lag(df, lag)
        if data is not None:
            # Merge and return the fields
            data_key = self.open_data(data, key)
            dfin = data_key.merge(df, how='left', on=key)
            dfin.index = data_key.index
            return(dfin[fields])
        else:
            # Return the entire dataset with keys
            return(df[key+fields])

    ###################
    # General Methods #
    ###################

    def volatility(self, fields, offset, min_periods, data, lag=0):
        """ Compute the volatility (standard deviation) of given fields.

        Args:
            fields (list): List of fields names.
                The fields may be fields presents wither in the 'fund' file
                or the 'names' file, or may be defined by one of the module
                function.
            offset (int or days): Maximum window of time to compute the
                volatility. Can be expressed in number of past observations or
                in time ('365D' for 365 days).

            min_periods (int): Minimum number of past observations to compute
                the volatility (cannot be expressed in time).
            lag (int, optional): If lag = N>0, returns the Nth lag of the
                fields. Can be positive (returns future values).
            data (str or pandas.DataFrame, optional): User provided data
                Required columns: [gvkey, datadate]

        Returns:
            pandas.Serie(s) or pandas.DataFrame:
                Series of thevolatility of the fields given.
                The index of the series returned corresponds to the
                index of the provided data.

        """
        key = self.key
        df = self.get_fields(key+fields)
        # Index the data with datadate
        df = df.set_index(self.col_date)
        g = df.groupby(self.col_id)[fields]
        # Compute the volatility
        std = g.rolling(offset, min_periods=min_periods).std(skipna=False)
        # Use float32 to save space
        std = std.astype('float32')
        std = std.reset_index()
        # Shift the values if lag required
        if lag != 0:
            stdl = std.groupby(self.col_id)[fields].shift(lag)
            std[fields] = stdl
        # Merge to the data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(std[key+fields], how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])

    ##########################
    # Variables Computations #
    ##########################

    def add_var(self, fn, fields=[]):
        """ Gives the ability to add a custom variable function """
        def _var(data):
            key = self.key
            df = self.get_fields(key+fields)
            df['v'] = fn(df)
            # Merge to the data
            dfu = self.open_data(data, key)
            dfin = dfu.merge(df[key+['v']], how='left', on=key)
            dfin.index = dfu.index
            return(dfin.v)
        setattr(self, fn.__name__, _var)

    #######################
    # Quarterly Variables #
    #######################

    ##### Redefinition of existing fields #####
    # Main purpose is to fill missing values for instance.

    def _xrdq(self, data):
        """ Return Expenses in R&D with 0 when missing values (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['xrdq']
        df[fields] = self._get_fund_fields(fields, df)  # Need the raw xrdq
        df.loc[df.xrdq.isna(), 'xrdq'] = 0
        return(df.xrdq)

    ##### Definition of new variables #####

    def _blevq(self, data):
        """ Return Book Leverage (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['ltq', 'txdbq', 'atq']
        df[fields] = self.get_fields(fields, data)
        # Replace txdbq by 0 when missing
        df.loc[df.txdbq.isna(), 'txdbq'] = 0
        df['blevq'] = (df.ltq - df.txdbq) / df.atq
        return(df.blevq)

    def _capxq(self, data):
        """ Capital Expenditure (Quarterly). """
        key = self.key
        fields = ['fyr', 'capxy']
        df = self.get_fields(fields)
        df['l1capxy'] = self.get_fields(['capxy'], df, lag=1)
        df.loc[df.capxy.isna(), 'capxy'] = 0
        df.loc[df.l1capxy.isna(), 'l1capxy'] = 0
        df['capxq'] = df.capxy - df.l1capxy
        df['month'] = pd.DatetimeIndex(df.datadate).month
        df['monthQ1'] = (df.fyr + 2) % 12 + 1
        condQ1 = df.month == df.monthQ1
        df.loc[condQ1, 'capxq'] = df[condQ1].capxy
        # Merge to the data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(df[key+['capxq']], how='left', on=key)
        dfin.index = dfu.index
        # Return the field
        return(dfin.capxq)

    def _dvtq(self, data):
        """ Return quarterly dividends.
        The quarterly dividend is the difference dvy[t]-dvy[t-1] except on
        the first quarter of the fiscal year where it is dvy[t].
        """
        key = self.key
        fields = ['fyr', 'dvy']
        df = self.get_fields(fields)
        df['l1dvy'] = self.get_fields(['dvy'], df, lag=1)
        df.loc[df.dvy.isna(), 'dvy'] = 0
        df.loc[df.l1dvy.isna(), 'l1dvy'] = 0
        df['dvtq'] = df.dvy - df.l1dvy
        df['month'] = pd.DatetimeIndex(df.datadate).month
        df['monthQ1'] = (df.fyr + 2) % 12 + 1
        condQ1 = df.month == df.monthQ1
        df.loc[condQ1, 'dvtq'] = df[condQ1].dvy
        # Merge to the data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(df[key+['dvtq']], how='left', on=key)
        dfin.index = dfu.index
        # Return the field
        return(dfin.dvtq)

    def _epsdq(self, data):
        """ Returns Diluted EPS. """
        key = self.key
        df = self.open_data(data, key)
        fields = ['ibq', 'cshfdq']
        df[fields] = self.get_fields(fields, data)
        df['epsd'] = df.ibq / df.cshfdq
        return(df.epsd)

    def _epsbq(self, data):
        """ Returns Basic EPS (non-diluted). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['ibq', 'cshprq']
        df[fields] = self.get_fields(fields, data)
        df['epsb'] = df.ibq / df.cshprq
        return(df.epsb)

    def _eqq(self, data):
        """ Return Equity (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['prccq', 'cshoq']
        df[fields] = self.get_fields(fields, data)
        df['eqq'] = df.prccq * df.cshoq
        return(df.eqq)

    def _hhiq(self, data):
        """ Return Herfindahl-Hirschman Index (Quarterly). """
        key = self.key
        fields = ['revtq', 'sic']
        df = self.get_fields(key+fields).dropna()
        gk = ['sic', 'datadate']
        # Compute market share
        df['revtqsum'] = df.groupby(gk).revtq.transform('sum')
        # ms = df.groupby(gk).revtq.sum().reset_index(name='revtqsum')
        # df = df.merge(ms, how='left', on=gk)
        df['s'] = (df.revtq / df.revtqsum) * 100
        # Compute HHI index
        df['s2'] = df.s**2
        df['hhiq'] = df.groupby(gk).s2.transform('sum')
        # This gives an index between 0 and 10,000.
        # Scale the index between 0 and 1
        df['hhiq'] = df.hhiq / 10000
        # hhi = df.groupby(gk).s2.sum().reset_index(name='hhiq')
        # df = df.merge(hhi, how='left', on=gk)
        # Merge to the data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(df[key+['hhiq']], how='left', on=key)
        dfin.index = dfu.index
        return(df.hhiq)

    def _litig(self, data):
        """ Litigation dummy.
        = 1 if SIC in [2833–2836, 8731–8734, 3570–3577, 3600–3674, 7371–7379,
                       5200–5961, 4812–4813, 4833, 4841, 4899, 4911, 4922–4924,
                       4931, 4941]
        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['sic']
        df[fields] = self.get_fields(fields, data)
        ranges = [range(2833, 2837), range(8731, 8735), range(3570, 3578),
                  range(3600, 3675), range(7371, 7380), range(5200, 5962),
                  range(4812, 4814), range(4833, 4834), range(4841, 4842),
                  range(4899, 4900), range(4911, 4912), range(4922, 4925),
                  range(4931, 4932), range(4941, 4942)]
        rs = []
        for r in ranges:
            for i in r:
                rs = rs + [i]
        # Remove missing values
        df = df.dropna()
        # Convert sic field to Integer
        #df.sic = df.sic.astype(int)
        # Create the litigation dummy
        df['litig'] = 0
        df.loc[df.sic.isin(rs), 'litig'] = 1
        return(df.litig.astype('Int32'))

    def _mb0q(self, data):
        """ Return Market-to-Book ratio 0 (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['eqq', 'ceqq']
        df[fields] = self.get_fields(fields, data)
        df['mb0q'] = df.eqq / df.ceqq
        return(df.mb0q)

    def _mb1q(self, data):
        """ Return Market-to-Book ratio 0 (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['eqq', 'ltq', 'atq']
        df[fields] = self.get_fields(fields, data)
        df['mb1q'] = (df.eqq + df.ltq) / df.atq
        return(df.mb1q)

    def _mroq(self, data):
        """ Return Operating margin (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['revtq', 'xoprq']
        df[fields] = self.get_fields(fields, data)
        df['mroq'] = df.revtq / df.xoprq
        df.loc[(df.mroq==np.inf) | (df.mroq==-np.inf), 'mroq'] = np.nan
        return(df.mroq)

    def _numfq(self, data):
        """ Return the number of firm by industry (SIC) (Quarterly). """
        key = self.key
        fields = ['sic']
        df = self.get_fields(key+fields).dropna()
        gk = ['sic', 'datadate']
        # Compute the number of gvkey by (sic, quarter)
        df['numfq'] = df.groupby(gk).gvkey.transform('count')
        # ms = df.groupby(gk).gvkey.count().reset_index(name='numfq')
        # df = df.merge(ms, how='left', on=gk)
        # Merge to the data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(df[key+['numfq']], how='left', on=key)
        dfin.index = dfu.index
        return(df.numfq.astype('Int32'))

    def _oaccq(self, data):
        """ Return Operating Accruals (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['actq', 'cheq', 'lctq', 'dlcq']
        l1f = ['l1'+f for f in fields]
        df[fields] = self.get_fields(fields, data)
        df[l1f] = self.get_fields(fields, data, lag=1)
        df['oaccq'] = ((df.actq - df.cheq) - (df.l1actq - df.l1cheq) -
                      ((df.lctq - df.dlcq) - (df.l1lctq - df.l1dlcq)))
        return(df.oaccq)

    def _roa0q(self, data):
        """ Return Return on Assets 0 (Quarterly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['niq', 'atq']
        df[fields] = self.get_fields(fields, data)
        df['roa0q'] = df.niq / df.atq
        return(df.roa0q)

    ####################
    # Yearly Variables #
    ####################

    ##### Redefinition of existing fields #####
    # Main purpose is to fill missing values for instance.


    def _at(self, data):
        """ Return Total Assets with missing valus when equals 0. """
        key = self.key
        df = self.open_data(data, key)
        fields = ['at']
        df[fields] = self._get_fund_fields(fields, df)
        df.loc[df['at']==0, 'at'] = np.nan
        return(df['at'])

    def _xrd(self, data):
        """ Return Expenses in R&D with 0 when missing values. """
        key = self.key
        df = self.open_data(data, key)
        fields = ['xrd']
        df[fields] = self._get_fund_fields(fields, df)  # Need the raw xrdq
        df.loc[df.xrd.isna(), 'xrd'] = 0
        return(df.xrd)

    def _xsga(self, data):
        """ Return Expenses in SGA with 0 when missing values. """
        key = self.key
        df = self.open_data(data, key)
        fields = ['xsga']
        df[fields] = self._get_fund_fields(fields, df)
        df.loc[df.xsga.isna(), 'xsga'] = 0
        return(df.xsga)

    def _tlcf(self, data):
        """ Return Tax Loss Carry Forward with 0 when missing values. """
        key = self.key
        df = self.open_data(data, key)
        fields = ['tlcf']
        df[fields] = self._get_fund_fields(fields, df)
        df.loc[df.tlcf.isna(), 'tlcf'] = 0
        return(df.tlcf)

    def _itcb(self, data):
        """ Return Investment Tax Credit with 0 when missing values. """
        key = self.key
        df = self.open_data(data, key)
        fields = ['itcb']
        df[fields] = self._get_fund_fields(fields, df)
        df.loc[df.itcb.isna(), 'itcb'] = 0
        return(df.itcb)

    ##### Definition of new variables #####

    def _act_lct(self, data):
        """ Return Current Ratio (Yearly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['act', 'lct']
        df[fields] = self.get_fields(fields, data)
        df['act_lct'] = df.act / df.lct
        return(df.act_lct)

    def _cfo(self, data):
        """ Cash flow from operations
        Equals oancf if fyear>=1987 and fopt - oacc if fyear<1987
        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['fyear', 'oancf', 'fopt', 'oacc']
        df[fields] = self.get_fields(fields, data)
        # Compute the before 1987 measure
        # set fopt=0 if not available
        df.loc[df.fopt.isnull(), 'fopt'] = 0.
        df['cfo'] = df.fopt - df.oacc
        # Complement with after 1987 measure
        # Only when there is a value after 1987
        cond = (df.fyear>=1987) & (df.oancf.notnull())
        df.loc[cond, 'cfo'] = df[cond].oancf
        return(df.cfo)

    def _eq(self, data):
        """ Equity (Yearly).
        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['prcc_f', 'csho']
        df[fields] = self.get_fields(fields, data)
        eq = df.prcc_f * df.csho
        return(eq)

    def _flev(self, data):
        """ Financial Leverage (Yearly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['dltt', 'dlc', 'eq']
        df[fields] = self.get_fields(fields, data)
        df['flev'] = (df.dltt + df.dlc) / df['eq']
        df.loc[df['eq']==0, 'flev'] = np.nan
        return(df.flev)

    def _mb(self, data):
        """ Market-to-Book ratio (Yearly). """
        key = self.key
        df = self.open_data(data, key)
        fields = ['eq', 'ceq']
        df[fields] = self.get_fields(fields, data)
        df['mb'] = df['eq'] / df.ceq
        df.loc[df.ceq==0, 'mb'] = np.nan
        return(df.mb)

    def _noacc(self, data):
        """ Non-operating accruals

        Non-operating accruals are:

        Net income before extraordinary items (ib)
        + Depreciation (dp)
        - Cash flow from operations (_cfo)
        - Operatig accruals (_oacc)

        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['ib', 'dp', 'cfo', 'oacc']
        df[fields] = self.get_fields(fields, data)
        df['noacc'] = df.ib + df.dp - df.cfo - df.oacc
        return(df.noacc)

    def _oacc(self, data):
        """ Operating Accruals

        Operating accruals are changes in non-cash assets (act-che)
        minus changes in current liabilities (lct-dlc) exluding short-term debt.

        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['act', 'che', 'lct', 'dlc']
        df[fields] = self.get_fields(fields, data)
        # Get the lag values of the fields
        l1fields = ['l1'+f for f in fields]
        df[l1fields] = self.get_fields(fields, data, lag=1)
        # Compute operating accruals
        df['oacc'] = ((df.act - df.che) - (df.l1act - df.l1che) -
                      ((df.lct - df.dlc) - (df.l1lct - df.l1dlc)))
        return(df.oacc)

    def _qual(self, data):
        """ Return qualified opinion dummy """
        key = self.key
        df = self.open_data(data, key)
        fields = ['auop']
        df[fields] = self.get_fields(fields, data)
        # Create the qual dummy
        df['qual'] = 0
        df.loc[df.auop=='2', 'qual'] = 1
        return(df.qual.astype('Int32'))

    def _roa(self, data):
        """ Return ROA ( net income / total assets) """
        key = self.key
        df = self.open_data(data, key)
        fields = ['ib', 'at']
        df[fields] = self.get_fields(fields, data)
        df['roa'] = df.ib / df['at']
        df.loc[df['at']==0, 'roa'] = np.nan
        return(df.roa)

    def _technology(self, data):
        """ Technology dummy.
        = 1 if SIC in [2830s, 3570s, 7370s, 8730s, 3825-3839]
        = 1 if SIC in [2833–2836, 8731–8734, 3570–3577, 3600–3674, 7371–7379,
                       5200–5961, 4812–4813, 4833, 4841, 4899, 4911, 4922–4924,
                       4931, 4941]
        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['sic']
        df[fields] = self.get_fields(fields, data)
        ranges = [range(2830, 2840), range(3570, 3580), range(7370, 7380),
                  range(8730, 8740), range(3825, 3840)]
        rs = []
        for r in ranges:
            for i in r:
                rs = rs + [i]
        # Remove missing values
        # ## sic field is text with 'None' when missing
        # #df.loc[df.sic=='None'] = np.nan
        df = df.dropna()
        # # Convert sic field to Integer
        # #df.sic = df.sic.astype(int)
        # Create the litigation dummy
        df['techno'] = 0
        df.loc[df.sic.isin(rs), 'techno'] = 1
        return(df.techno.astype('Int32'))
