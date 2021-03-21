"""
Provides processing functions for IBES data
"""

import pandas as pd
# import pywrds as pw
# import pywrds.crspcomp as crspcomp
# import numpy as np
# from fuzzywuzzy import fuzz
from rapidfuzz import fuzz
from pilates import wrds_module
from pandas.tseries.offsets import MonthEnd


class ibes(wrds_module):

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.scores = [0, 1, 4]
        self.col_ticker = 'ticker'
        self.col_fpedats = 'fpedats'
        # Default filters
        self.usfirm = 1

    def set_ticker_column(self, col_ticker):
        """ Set the name of the column to use for the IBE ticker.
        """
        self.col_ticker = col_ticker

    def set_fpedats_column(self, col_fpedats):
        """ Set the column (date) to use as the Forecast Period End Date for
        the user data.
        """
        self.col_fpedats = col_fpedats

    def set_unadjust(self, unadjust):
        """ If set at True, all the values will be unadjusted.
        (The detail and guidance files must be adjusted).
        """
        self.unadjust = unadjust

    def set_guidance_pdicity(self, pdicity):
        if pdicity in ['QTR', 'qtr', 'Q', 'q', 'Quarterly', 'quarterly']:
            self.guidance_pdicity = 'QTR'
        elif pdicity in ['ANN', 'ann', 'A', 'a', 'Annual', 'annual']:
            self.guidance_pdicity = 'ANN'
        elif pdicity in ['SAN', 'san', 'SA', 'sa', 'Semiannual', 'semiannual']:
            self.guidance_pdicity = 'SAN'
        else:
            raise Exception('IBES guidance pdicity must be either ' +
                            'QTR, ANN or SAN')

    def set_guidance_units(self, units):
        if units in ['millions', 'P/S', '%', 'billions', 'P/1000S']:
            self.guidance_units = units
        else:
            raise Exception('IBES guidance units must be either ' +
                            "'millions', 'P/S', '%', 'billions' or 'P/1000S'")

    def set_measure(self, measure):
        """ Set the measure to use.
        """
        self.measure = measure

    def set_forecasts_security(self, pdf):
        if pdf in ['P', 'D', 'DP', 'All']:
            self.pdf = pdf
        else:
            raise Exception('The security to use for the forecasts is either' +
                            'P: Use only Primary, ' +
                            'D: Use only Diluted, ' +
                            'DP: Use Diluted first and Primary ' +
                            'if no Diluted available, ' +
                            'All: Does not filter on the type of security')

    def set_forecasts_pdicity(self, pdicity):
        if pdicity in ['QTR', 'qtr', 'Q', 'q', 'Quarterly', 'quarterly']:
            self.fpis = ['6', '7', '8', '9', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                         'L', 'Y']
        elif pdicity in ['ANN', 'ann', 'A', 'a', 'Annual', 'annual',
                         'Y', 'y', 'Yearly', 'yearly']:
            self.fpis = ['1', '2', '3', '4', 'E', 'F', 'G', 'H', 'I', 'U', 'X']
        elif pdicity in ['SAN', 'san', 'SA', 'sa', 'Semiannual', 'semiannual']:
            self.fpis = ['A', 'B', 'C', 'D', 'J', 'K', 'Z']
        elif pdicity in ['L', 'LT', 'Long-term', 'long-term']:
            self.fpis = ['0']
        else:
            raise Exception('The Forecast Periodicity  should be ' +
                            'either QTR, ANN, SAN, or LT.')

    def _ibes_crsp_linktable(self):
        """ This function returns the ICLINK table that links IBES and CRSP.
        The code is adapted from Qingyi (Freda) Song Drechsler
        (updated June 2020)
        https://www.fredasongdrechsler.com/python-code/iclink
        """
        #########################
        # Step 1: Link by CUSIP #
        #########################
        # 1.1 IBES: Get the list of IBES Tickers for US firms in IBES
        cols = ['ticker', 'cusip', 'cname', 'sdates']
        _ibes1 = self.open_data(self.id, cols)
        # Create first and last 'start dates' for a given cusip
        # Use agg min and max to find the first and last date per group
        # then rename to fdate and ldate respectively
        g = _ibes1.groupby(['ticker', 'cusip'])
        _ibes1['fdate'] = g.sdates.transform('min')
        _ibes1['ldate'] = g.sdates.transform('max')
        # keep only the most recent company name
        # determined by having sdates = ldate
        _ibes2 = _ibes1[_ibes1.sdates == _ibes1.ldate].drop(['sdates'], axis=1)
        # 1.2 CRSP: Get all permno-ncusip combinations
        cols = ['permno', 'ncusip', 'comnam', 'namedt', 'nameenddt']
        _crsp1 = self.open_data(self.d.crsp.stocknames, cols)
        g = _crsp1.groupby(['permno', 'ncusip'])
        # First namedt and last nameenddt
        _crsp1['fnamedt'] = g.namedt.transform('min')
        _crsp1['lnameenddt'] = g.nameenddt.transform('max')
        # Keep only most recent company name
        _crsp2 = _crsp1[_crsp1.nameenddt == _crsp1.lnameenddt]
        # Drop and Rename
        _crsp2 = _crsp2.drop(['namedt', 'nameenddt'], axis=1)
        _crsp2 = _crsp2.rename(columns={'fnamedt':'namedt',
                                        'lnameenddt':'nameenddt'})
        # 1.3 Create CUSIP Link Table
        # Link by full cusip, company names and dates
        _link1_1 = pd.merge(_ibes2, _crsp2, how='inner', left_on='cusip',
                            right_on='ncusip')
        # Keep link with most recent company name
        g = _link1_1.groupby(['ticker', 'permno'])
        _link1_1['lldate'] = g.ldate.transform('max')
        _link1_2 = _link1_1[_link1_1.ldate == _link1_1.lldate]
        # Calculate name matching ratio using FuzzyWuzzy
        # Note: fuzz ratio = 100 -> match perfectly
        #       fuzz ratio = 0   -> do not match at all
        # Comment: token_set_ratio is more flexible in matching the strings:
        # fuzz.token_set_ratio('AMAZON.COM INC',  'AMAZON COM INC')
        # returns value of 100
        # fuzz.ratio('AMAZON.COM INC',  'AMAZON COM INC')
        # returns value of 93
        _link1_2['name_ratio'] = _link1_2\
            .apply(lambda x: fuzz.token_set_ratio(x.comnam, x.cname), axis=1)
        # Note on parameters:
        # The following parameters are chosen to mimic the SAS macro %iclink
        # In %iclink, name_dist < 30 is assigned score = 0
        # where name_dist=30 is roughly 90% percentile in total distribution
        # and higher name_dist means more different names.
        # In name_ratio, I mimic this by choosing 10% percentile as cutoff to
        # assign score = 0
        # 10% percentile of the company name distance
        name_ratio_p10 = _link1_2.name_ratio.quantile(0.10)
        # Function to assign score for companies matched by:
        # full cusip and passing name_ratio
        # or meeting date range requirement
        # assign size portfolio
        _link1_2['score'] = 3
        _link1_2.loc[_link1_2.name_ratio >= name_ratio_p10, 'score'] = 2
        _link1_2.loc[(_link1_2.fdate <= _link1_2.nameenddt) &
                     (_link1_2.ldate >= _link1_2.namedt), 'score'] = 1
        _link1_2.loc[(_link1_2.fdate <= _link1_2.nameenddt) &
                     (_link1_2.ldate >= _link1_2.namedt) &
                     (_link1_2.name_ratio >= name_ratio_p10), 'score'] = 0
        _link1_2 = _link1_2[['ticker', 'permno', 'cname', 'comnam',
                             'name_ratio', 'score']]
        _link1_2 = _link1_2.drop_duplicates()

        ##########################
        # Step 2: Link by TICKER #
        ##########################
        # Find links for the remaining unmatched cases using Exchange Ticker
        # Identify remaining unmatched cases
        _nomatch1 = pd.merge(_ibes2[['ticker']],
                             _link1_2[['permno', 'ticker']],
                             on='ticker', how='left')
        _nomatch1 = _nomatch1.loc[_nomatch1.permno.isnull()]\
            .drop(['permno'], axis=1).drop_duplicates()
        # Add IBES identifying information
        cols = ['ticker', 'cname', 'oftic', 'sdates', 'cusip']
        ibesid = self.open_data(self.id, cols)
        ibesid = ibesid.loc[ibesid.oftic.notna()]
        _nomatch2 = pd.merge(_nomatch1, ibesid, how='inner', on=['ticker'])
        # Create first and last 'start dates' for Exchange Tickers
        # Label date range variables and keep only most recent company name
        g = _nomatch2.groupby(['ticker', 'oftic'])
        _nomatch2['fdate'] = g.sdates.transform('min')
        _nomatch2['ldate'] = g.sdates.transform('max')
        _nomatch3 = _nomatch2[_nomatch2.sdates == _nomatch2.ldate]
        # Get entire list of CRSP stocks with Exchange Ticker information
        cols = ['ticker', 'comnam', 'permno', 'ncusip', 'namedt', 'nameenddt']
        _crsp_n1 = self.open_data(self.d.crsp.stocknames, cols)
        _crsp_n1 = _crsp_n1[_crsp_n1.ticker.notna()]
        # Arrange effective dates for link by Exchange Ticker
        g = _crsp_n1.groupby(['permno', 'ticker'])
        _crsp_n1['fnamedt'] = g.namedt.transform('min')
        _crsp_n1['lnameenddt'] = g.nameenddt.transform('min')
        _crsp_n2 = _crsp_n1[_crsp_n1.nameenddt == _crsp_n1.lnameenddt]
        _crsp_n2 = _crsp_n2.drop(['namedt', 'nameenddt'], axis=1)
        _crsp_n2 = _crsp_n2.rename(columns={'fnamedt':'namedt',
                                          'lnameenddt':'nameenddt',
                                          'ticker': 'crsp_ticker'})
        # Merge remaining unmatched cases using Exchange Ticker
        # Note: Use ticker date ranges as exchange tickers are reused overtime
        _link2_1 = pd.merge(_nomatch3, _crsp_n2, how='inner',
                            left_on=['oftic'], right_on=['crsp_ticker'])
        _link2_1 = _link2_1.loc[(_link2_1.ldate >= _link2_1.namedt) &
                                (_link2_1.fdate <= _link2_1.nameenddt)]
        # Score using company name using 6-digit CUSIP
        # and company name spelling distance
        _link2_1['name_ratio'] = _link2_1\
            .apply(lambda x: fuzz.token_set_ratio(x.comnam, x.cname), axis=1)
        _link2_2 = _link2_1
        _link2_2['cusip6'] = _link2_2.cusip.str.slice(stop=6)
        _link2_2['ncusip6'] = _link2_2.ncusip.str.slice(stop=6)
        # Score using company name using 6-digit CUSIP
        # and company name spelling distance
        _link2_2['score'] = 6
        _link2_2.loc[_link2_2.name_ratio >= name_ratio_p10, 'score'] = 5
        _link2_2.loc[_link2_2.cusip6 == _link2_2.ncusip6, 'score'] = 4
        _link2_2.loc[(_link2_2.cusip6 == _link2_2.ncusip6) &
                     (_link2_2.name_ratio >= name_ratio_p10), 'score'] = 0
        # assign size portfolio
        # Some companies may have more than one TICKER-PERMNO link
        # so re-sort and keep the case (PERMNO & Company name from CRSP)
        # that gives the lowest score for each IBES TICKER
        _link2_2 = _link2_2[['ticker', 'permno', 'cname', 'comnam',
                             'name_ratio', 'score']]
        g = _link2_2.groupby(['ticker'])
        _link2_2['min_score'] = g.score.transform('min')
        _link2_3 = _link2_2[_link2_2.score == _link2_2.min_score]
        _link2_3 = _link2_3[['ticker', 'permno', 'cname', 'comnam',
                             'score']].drop_duplicates()

        #####################################
        # Step 3: Finalize Links and Scores #
        #####################################
        iclink = _link1_2.append(_link2_3)
        return(iclink)

    def _adjustmentfactors_table(self):
        """ Create an adjustment factor table with a start and end date
        for each factor.
        """
        # Add the necessary columns to process the data
        cols = ['ticker', 'spdates', 'adj']
        # Open adjustment factors data
        df_adj = self.open_data(self.adj, cols)
        # Select the firms that have at least one split
        c = df_adj.groupby('ticker').count()['adj'].reset_index()
        t = c[c.adj > 1]['ticker']
        df = df_adj[df_adj.ticker.isin(t)].sort_values(['ticker', 'spdates'])
        # Remove the first date as it does not coincide with the start of the
        # sample and create the start date
        c = df.groupby('ticker')
        # df['spdatesrank'] = c.spdates.rank()
        df['min_spdates'] = c.spdates.transform('min')
        df['startdt'] = df.spdates
        df.loc[df.min_spdates == df.startdt, 'startdt'] = pd.NaT
        # Create the end date
        df['enddt'] = c.spdates.shift(-1)
        # Drop unecessary columns and return the adjustment table
        df = df.drop(['spdates', 'min_spdates'], axis=1)
        return(df)

    def _get_adjustment_factors(self, data):
        """ Get the adjustment factors for a given DataFrame.
        Arguments:
            df --   User provided dataset.
                    Required columns: [ticker, fpedats]
        This function requires the following object attributes:
            self.adj -- Data from the IBES 'det_adj' file (adjustments file)
        """
        key = ['ticker', 'fpedats']
        dfu = self.open_data(data, key).drop_duplicates()
        adjfac = self._adjustmentfactors_table()
        # Add all the fpedats to all the tickers in the adjustment file
        # Ideally we would like to do a conditional merge
        m = adjfac.merge(dfu, how='left', on='ticker')
        # Remove the dates that are not correct and the unecessary columns
        m = m[m.fpedats >= m.startdt]
        m = m[m.fpedats < m.enddt]
        m = m.drop(['startdt', 'enddt'], axis=1)
        # Merge the adjustment factors to the user data
        dfu = self.open_data(data, key)
        index = dfu.index
        dfin = dfu.merge(m, how='left', on=key)
        # Replace null by 1 (adjustment factors)
        dfin.loc[dfin.adj.isna(), 'adj'] = 1
        # Make sure index is correct and return
        dfin.index = index
        return(dfin.adj)

    def _unadjust(self, fields, data):
        """ Unajust the fields from the data.
        Only used internally.
        """
        # Get the adjustment factors
        data['adj'] = self._get_adjustment_factors(data)
        # Adjust the analyst forecasts
        data[fields] = data[fields].multiply(data.adj, axis='index')
        return(data[fields].astype('float32'))

    def ibesticker_from_gvkey(self, data):
        """ Returns IBES ticker from COMPUSTAT gvkey.
        Arguments:
            data -- User provided data.
                    Required columns: [gvkey, datadate]
        """
        # Add the permno
        key = ['gvkey', 'datadate']
        df = self.open_data(data, key).drop_duplicates().dropna()
        df['permno'] = self.d.crsp.permno_from_gvkey(df)
        # Add the IBES ticker
        # Get the IBES-CRSP link table
        iclink = self._ibes_crsp_linktable()
        iclink = iclink[iclink.score.isin(self.scores)]
        # We cannot optain a 1-1 match between IBES and CRSP as IBES ticker
        # is a permanent identifier but permno is not
        # We need to find the best IBES-COMPUSTAT match as gvkey is permanent
        # Merge datasets and then clean
        # For each [gvkey, datadate], we should have only 1 IBES ticker
        d = df.merge(iclink, how='left', on=['permno'])
        d = d[['gvkey', 'datadate', 'permno', 'ticker', 'name_ratio', 'score']]
        d = d.drop_duplicates()
        dup = d[key].duplicated(keep=False)
        nd0 = d[~dup][key+['ticker']]
        d_dup = d[dup].sort_values(key)
        # Clean the duplicates
        # 1. If given the choice, keep the matches without name_ratio or
        # the ones with the highest name ratio
        g = d_dup.groupby(key)
        d_dup['maxnr'] = g.name_ratio.transform(pd.Series.max, skipna=False)
        d_dup = d_dup[(d_dup.maxnr==d_dup.name_ratio) |
                      (d_dup.maxnr.isna() & d_dup.name_ratio.isna())]
        dup = d_dup[key].duplicated(keep=False)
        nd1 = d_dup[~dup][key+['ticker']]
        d_dup = d_dup[dup].sort_values(key)
        # 3. Keep the first ticker in alphabetical order (random)
        # Arbitrary but get rid of all duplicates
        g = d_dup.groupby(key)
        d_dup['first_t'] = g.ticker.transform('min')
        d_dup = d_dup[d_dup.ticker == d_dup.first_t]
        nd2 = d_dup
        # Concatenate the matches
        match = pd.concat([nd0, nd1, nd2])
        # We want to find 1-1 relationship between gvkey and ticker
        dup = match[['gvkey', 'datadate']].duplicated(keep=False)
        n_dup = len(match[dup])
        if n_dup > 0:
            print("Warning: The merged ticker",
                  "contains {:} duplicates".format(n_dup))
        # Merge with user data and return
        dfu = self.open_data(data, key)
        dfin = dfu.merge(match, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.ticker)

    def get_fpedats(self, data):
        """ Determine the forecast period end date using year and month
        from the col_fpedats field from user data.
        Arguments:
            data -- User data.
                    Required columns: [ticker, 'col_fpedats']
        """
        df = self.open_data(self.det, ['ticker',
                                         'fpedats']).drop_duplicates()
        dt = pd.to_datetime(df.fpedats).dt
        df['yr'] = dt.year
        df['mon'] = dt.month
        dfu = self.open_data(data, ['ticker', self.col_fpedats])
        dt = pd.to_datetime(dfu[self.col_fpedats]).dt
        dfu['yr'] = dt.year
        dfu['mon'] = dt.month
        key = ['ticker', 'yr', 'mon']
        dfin = dfu.merge(df, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.fpedats)

    def _get_fields_det(self, fields=None):
        """ Return the fields from the det file filtered. """
        key = ['ticker', 'fpedats']
        # Filter measure
        det = self.open_data(self.det, ['measure'])
        kmeasure = det.measure == self.measure
        # Filter usfirm
        det = self.open_data(self.det, ['usfirm'])
        kusfirm = det.usfirm == self.usfirm
        # Filter fpi
        det = self.open_data(self.det, ['fpi'])
        kfpi = det.fpi.isin(self.fpis)
        # Filter pdf (Primary / Diluted)
        if self.pdf == 'All':
            kpdf = True
        elif self.pdf == 'P' or self.pdf == 'D':
            det = self.open_data(self.det, ['pdf'])
            kpdf = det.pdf == self.pdf
        elif self.pdf == 'DP':
            det = self.open_data(self.det, key+['pdf'])
            pdfd = det.pdf == 'D'
            pdfp = det.pdf == 'P'
            detk = det[key].drop_duplicates()
            detd = det[pdfd].drop_duplicates()
            detd = detd.rename({'pdf': 'pdfd'}, axis=1)
            detp = det[pdfp].drop_duplicates()
            detp = detp.rename({'pdf': 'pdfp'}, axis=1)
            detk = detk.merge(detd, how='left', on=key)
            detk = detk.merge(detp, how='left', on=key)
            detk['PnotD'] = detk.pdfd.isna() & detk.pdfp.notna()
            # Keep the observations where pdf=='D' and where the key is
            # in detk[detk.PnotD]
            indexdet = det.index
            det = det.merge(detk[key+['PnotD']], how='left', on=key)
            det.index = indexdet
            kpdf = (det.pdf == 'D') | det.PnotD
        # Create the final filter
        mask = kmeasure & kusfirm & kfpi & kpdf
        det = self.open_data(self.det, fields)
        det = det[mask].drop_duplicates()
        # Unadjust the values if needed (value, actual)
        if fields is not None and self.unadjust:
            for f in ['value', 'actual']:
                if f in fields:
                    det[f] = self._unadjust([f], det)
        return(det)

    def _get_fields_guidance(self, fields=None):
        """ Return the fields from the guidance file filtered. """
        cols = ['pdicity', 'measure', 'usfirm', 'units', 'prd_yr', 'prd_mon']
        df = self.open_data(self.guidance, cols+fields)
        # Keep quarterly EPS forecasts for US firms
        df = df[(df.pdicity == self.guidance_pdicity) &
                (df.measure == self.measure) &
                (df.usfirm == self.usfirm) &
                (df.units == self.guidance_units)]
        # Unadjust the values if needed (value, actual)
        if fields is not None and self.unadjust:
            # Add a synthetic fpedats using end of month to get the adjustments
            df['ym'] = (df.prd_yr*100 + df.prd_mon).astype(int)
            df['fpedats'] = pd.to_datetime(df.ym, format='%Y%m') + MonthEnd()
            for f in ['val_1', 'val_2', 'mean_at_date']:
                if f in fields:
                    df[f] = self._unadjust([f], df)
        return(df[fields])

    def _get_fields_ptg(self, fields=None):
        """ Return the fields from the ptg (price targets) file filtered. """
        None

    def get_management_forecasts(self, data, fields, which='last',
                                 nadays=None):
        """ Return the management forecasts for a given forecast period.
        Keep the forecasts from US firms that lie between
        the previous earnings annoucement and the forecast period end date.
        Arguments:
            data -- User provided data
                    Required columns:   [ticker, 'col_fpedats']
            fields -- Fields required from the guidance file
            which --    Defines which MF to keep if multiple ('last', 'first')
                        If None, return all management forecasts.
            fptype --   Forecast Period Type ('Q' for quarterly,
                        'Y' for annual, 'S' for semiannual,
                        'L' for long-term growth)
            for_quarter --  Integer specifying the quarter for which the
                            consensus should relate to.
                            'for_quarter=0' returns the consensus
                            for the current forecast period end.
                            'for_quarter=N' returns the consensus for
                            the Nth quarter after the current forecast
                            period end.
        This function requires the following object attributes:
            self.guidance -- Data from the IBES guidance data.
                             Required columns:
                                 [ticker, pdicity, measure, usfirm, units,
                                 anndats, anntims, prd_yr, prd_mon,
                                 actdats, acttims]
            self.det -- Data from an IBES detail files (for instance
                        the 'det_epsus' file (analysts forecasts for US))
                        Required columns: [ticker, fpedats, anndats_act]
            self.measure -- Measure to use to compute the consensus
                            ('EPS', 'BPS', ...)
            self.pdf -- Type of security used for the forecast.
                        pdf='P': Use only Primary
                        pdf='D': Use only Diluted
                        pdf='DP': Use Diluted and Primary if
                        no Diluted available
                        pdf='All': Use all forecasts
            self.pdicity -- Periodicity of the forecast ('QTR', 'ANN', SAN')
        """
        cols = ['ticker', 'anndats', 'anntims',
                'prd_yr', 'prd_mon', 'actdats', 'acttims']
        fields_guid = [f for f in fields if f not in cols]
        df = self._get_fields_guidance(cols+fields_guid)
        # Get the previous earnings annoucement
        df['eal1'] = self.get_ea_dates(df, shift=-1, nadays=nadays)
        # Compute the max fpedats (not available in guidance)
        df['ym'] = (df.prd_yr*100 + df.prd_mon).astype(int)
        df['fpedats_max'] = pd.to_datetime(df.ym, format='%Y%m') + MonthEnd()
        df['fpedats_max'] = df.fpedats_max.dt.date
        # Keep the managements forecasts between EA(t-1) and FPE(t)
        df = df[(df.anndats >= df.eal1) & (df.anndats <= df.fpedats_max)]
        # Select the first or last MF if multiple
        df.anntims = pd.to_timedelta(df.anntims, unit='s')
        df['time'] = pd.to_datetime(df.anndats) + df.anntims
        key = ['ticker', 'prd_yr', 'prd_mon']
        if which == 'last':
            df['time_keep'] = df.groupby(key).time.transform('max')
        elif which == 'first':
            df['time_keep'] = df.groupby(key).time.transform('min')
        else:
            raise Exception("Argument 'which' must be either " +
                            "'first' or 'last'")
        df = df[df.time == df.time_keep]
        # Keep the latest activation time if more dupicates
        df.acttims = pd.to_timedelta(df.acttims, unit='s')
        df['act_time'] = pd.to_datetime(df.actdats) + df.acttims
        df['act_time_keep'] = df.groupby(key).act_time.transform('max')
        df = df[df.act_time == df.act_time_keep]
        # Merge with user data and return the fields
        data_cols = self.d.get_fields_names(data)
        key = ['ticker', 'prd_yr', 'prd_mon']
        # Prepare the user data for merge
        if self.col_fpedats in data_cols:
            dfu = self.open_data(data, ['ticker', self.col_fpedats])
            dt = pd.to_datetime(dfu[self.col_fpedats]).dt
            dfu['prd_yr'] = dt.year
            dfu['prd_mon'] = dt.month
        elif 'fpedats' in data_cols:
            dfu = self.open_data(data, ['ticker', 'fpedats'])
            dt = pd.to_datetime(dfu.fpedats).dt
            dfu['prd_yr'] = dt.year
            dfu['prd_mon'] = dt.month
        else:  # For use when using guidance data
            dfu = self.open_data(data, key)
        dfin = dfu.merge(df[key+fields], how='left', on=key)
        dfin.index = dfu.index
        # Return the data
        return(dfin[fields])

    def _get_ea_fields(self, fields):
        """ Return fields for the earnings annoucements.
        Uses the IBES detail file and clean the earnings annoucements.
        """
        # Get the earnings annoucements dates
        cols = ['ticker', 'fpedats', 'anndats_act', 'anntims_act',
                'actdats_act', 'acttims_act']
        # ea = self.open_data(self.det, cols)
        fields_det = [f for f in fields if f not in cols]
        ea = self._get_fields_det(cols+fields_det)
        ea = ea[ea.anndats_act.notna()].drop_duplicates()
        # Clean the earnings annoucements
        # Keep earnings annoucements occuring after
        # the forecast period end date.
        ea = ea[ea.anndats_act > ea.fpedats]
        # Keep the earliest earnings annoucements when multiple
        key = ['ticker', 'fpedats']
        ea.acttims_act = pd.to_timedelta(ea.acttims_act, unit='s')
        ea['act_time'] = pd.to_datetime(ea.actdats_act) + ea.acttims_act
        ea['first_act_time'] = ea.groupby(key).act_time.transform('min')
        ea = ea[ea.act_time == ea.first_act_time]
        # Return the fields
        return(ea[fields])

    def get_ea_fields(self, fields, data, shift=0):
        """ Return the earnings annoucement fields for a given
        forecast period end.
        Arguments:
            fields -- Fields to return
            data --   DataFrame for which the earnings annoucements are needed.
                    Required columns:   [ticker, 'col_fpedats']
                                    or  [ticker, prd_yr, prd_mon]
            shift --    Return previous or future earnings annoucement dates.
                        If shift=-N, returns the Nth previous annoucement date.
                        If shift=+N, returns the Nth future annoucement date.
        This function requires the following object attributes:
            self.det -- Data from an IBES detail files (for instance
                        the 'det_epsus' file (analysts forecasts for US))
                        Required columns: [ticker, fpedats, anndats_act]
        """
        # Get the earnings annoucements dates
        key = ['ticker', 'fpedats']
        ea = self._get_ea_fields(key+fields)
        # Shift the values when required
        if shift != 0:
            g = ea.sort_values(key).groupby('ticker')
            eas = g[fields].shift(-shift)
            ea[fields] = eas
        # Merge the EA dates to the user data
        # Determine which key to use
        data_cols = self.d.get_fields_names(data)
        key = ['ticker', 'fpedats']
        if self.col_fpedats in data_cols:
            dfu = self.open_data(data, ['ticker', self.col_fpedats])
            dfu['fpedats'] = dfu[self.col_fpedats]
        elif 'fpedats' in data_cols:
            dfu = self.open_data(data, key)
        else:  # For use when using guidance data
            key = ['ticker', 'prd_yr', 'prd_mon']
            dfu = self.open_data(data, key)
            dt = pd.to_datetime(ea.fpedats).dt
            ea['prd_yr'] = dt.year
            ea['prd_mon'] = dt.month
        dfin = dfu.merge(ea[key+fields], how='left', on=key)
        dfin.index = dfu.index
        return(dfin[fields])

    def get_ea_dates(self, data, shift=0, nadays=None):
        """ Return the earnings annoucement dates for a given
        forecast period end.
        Arguments:
            data --   DataFrame for which the earnings annoucements are needed.
                    Required columns:   [ticker, 'col_fpedats']
                                    or  [ticker, prd_yr, prd_mon]
            shift --    Return previous or future earnings annoucement dates.
                        If shift=-N, returns the Nth previous annoucement date.
                        If shift=+N, returns the Nth future annoucement date.
            nadays --   When shift != 0, set the number of days to create
                        a 'fake' annoucement date (for instance, if shift=-1
                        nadays=120, set the annoucement date 120 days
                        before the current one, if shift=-2, set the
                        announcement date 240 days before).
        This function requires the following object attributes:
            self.det -- Data from an IBES detail files (for instance
                        the 'det_epsus' file (analysts forecasts for US))
                        Required columns: [ticker, fpedats, anndats_act]
        Returns:
            Series of earnings annoucements dates with data index.
        """
        # Get the earnings annoucements dates
        key = ['ticker', 'fpedats']
        ea = self._get_ea_fields(key+['anndats_act'])
        # Shift the earnings annoucement dates when required
        if shift != 0:
            g = ea.sort_values(key).groupby('ticker')
            eas = g['anndats_act'].shift(-shift)
            ea['eas'] = eas
            if nadays is not None:
                neas = ea.eas.isna()
                ea.loc[neas, 'eas'] = (ea[neas].anndats_act -
                                       pd.Timedelta(str(-shift * nadays) +
                                                    'days'))
            ea.anndats_act = ea.eas
        # Merge the EA dates to the user data
        # Determine which key to use
        data_cols = self.d.get_fields_names(data)
        key = ['ticker', 'fpedats']
        if self.col_fpedats in data_cols:
            dfu = self.open_data(data, ['ticker', self.col_fpedats])
            dfu['fpedats'] = dfu[self.col_fpedats]
        elif 'fpedats' in data_cols:
            dfu = self.open_data(data, key)
        else:  # For use when using guidance data
            key = ['ticker', 'prd_yr', 'prd_mon']
            dfu = self.open_data(data, key)
            dt = pd.to_datetime(ea.fpedats).dt
            ea['prd_yr'] = dt.year
            ea['prd_mon'] = dt.month
        dfin = dfu.merge(ea[key+['anndats_act']], how='left', on=key)
        dfin.index = dfu.index
        return(dfin.anndats_act)

    def get_ea_values(self, data, shift=0):
        """ Return the earnings annoucement dates for a given
        forecast period end.
        Arguments:
            data --   DataFrame for which the earnings annoucements are needed.
                    Required columns:   [ticker, 'col_fpedats']
                                    or  [ticker, prd_yr, prd_mon]
            shift --    Return previous or future earnings annoucement dates.
                        If shift=-N, returns the Nth previous annoucement date.
                        If shift=+N, returns the Nth future annoucement date.
        This function requires the following object attributes:
            self.det -- Data from an IBES detail files (for instance
                        the 'det_epsus' file (analysts forecasts for US))
                        Required columns: [ticker, fpedats, anndats_act]
        Returns:
            Series of earnings annoucements dates with data index.
        """
        return(self.get_ea_fields(['actual'], data, shift))
        # # Get the earnings annoucements dates
        # key = ['ticker', 'fpedats']
        # ea = self._get_ea_fields(key+['actual'])
        # # Shift the values when required
        # if shift != 0:
        #     g = ea.sort_values(key).groupby('ticker')
        #     eas = g['actual'].shift(-shift)
        #     ea['eas'] = eas
        #     ea.actual = ea.eas
        # # Merge the EA dates to the user data
        # # Determine which key to use
        # data_cols = self.d.get_fields_names(data)
        # key = ['ticker', 'fpedats']
        # if self.col_fpedats in data_cols:
        #     dfu = self.open_data(data, ['ticker', self.col_fpedats])
        #     dfu['fpedats'] = dfu[self.col_fpedats]
        # elif 'fpedats' in data_cols:
        #     dfu = self.open_data(data, key)
        # else:  # For use when using guidance data
        #     key = ['ticker', 'prd_yr', 'prd_mon']
        #     dfu = self.open_data(data, key)
        #     ea['prd_yr'] = ea.fpedats.dt.year
        #     ea['prd_mon'] = ea.fpedats.dt.month
        # dfin = dfu.merge(ea[key+['actual']], how='left', on=key)
        # dfin.index = dfu.index
        # return(dfin.actual)

    def get_consensus(self, data, startdt, enddt, for_quarter=0, n=5,
                      stat='mean'):
        """ Return analysts forecasts from a given period for a given quarter.
        Arguments:
            data -- User provided dataset
                    Required columns: [ticker, fpedats, 'startdt', 'enddt']
            startdt --  Beginning of the period used to compute the consensus.
            enddt --    End of the period used to compute the consensus.
            for_quarter --  Integer specifying the quarter for which the
                            consensus should relate to.
                            'for_quarter=0' returns the consensus
                            for the current forecast period end.
                            'for_quarter=N' returns the consensus for
                            the Nth quarter after the current forecast
                            period end.
            n --    Number of analysts forecasts to compte the consensus
            stat --   Statistic to compute the consensus ('mean', 'median').
        This function requires the following object attributes:
            self.det -- Data from an IBES detail files (for instance
                        the 'det_epsus' file (analysts forecasts for US))
                        Required columns:   [ticker, fpedats, analys, value,
                                             anndats, anntims, measure, fpi,
                                             pdf, usfirm]
            self.adj -- Data from the IBES 'det_adj' file (adjustments file)
            self.usfirm --  1 if use only US firms (default 1)
        """
        key = ['ticker', 'fpedats']
        cols = ['ticker', 'fpedats', 'analys', 'value', 'anndats', 'anntims']
        det = self._get_fields_det(cols)
        # # Get the adjustment factors
        # det['adj'] = self._get_adjustment_factors(det)
        # # Adjust the analyst forecasts
        # det['value'] = det.value * det.adj
        # Get the fpedats for which the consensus should be computed
        detu = det[['ticker', 'fpedats']].drop_duplicates().sort_values(key)
        detu['fpec'] = detu.groupby('ticker').fpedats.shift(-for_quarter)
        fpec = detu[['ticker', 'fpedats', 'fpec']]
        # Get the start and end date to compute the
        # consensus by (ticker, fpedats).
        key = ['ticker', self.col_fpedats]
        cdt = self.open_data(data, key + [startdt, enddt]).drop_duplicates()
        # Keep the rows for which a consensus can be computed
        cdt = cdt.dropna()
        # Add the fpedats for which the consensus should be computed
        cdt.loc[:, 'fpedats'] = cdt[self.col_fpedats]
        cdt = cdt.merge(fpec, how='left', on=['ticker', 'fpedats'])
        # Merge the datasets to compute the consensus
        det = det.rename({'fpedats': 'fpec'}, axis=1)
        key = ['ticker', 'fpec']
        df = cdt.merge(det, how='left', on=key)
        # Keep the analysts forecasts to consider
        df = df[(df.anndats >= df[startdt]) & (df.anndats <= df[enddt])]
        # Keep the last n analysts forecasts
        df.anntims = pd.to_timedelta(df.anntims, unit='s')
        df['time'] = pd.to_datetime(df.anndats) + df.anntims
        df = df.drop(['anndats', 'anntims'], axis=1)
        df = df.sort_values(['ticker', 'fpec', 'time'])
        df = df.groupby(key).head(n+1)
        # Select the correct statistical function
        fn = getattr(pd.core.groupby.GroupBy, stat)
        # Compute the consensus
        key = ['ticker', 'fpedats', 'fpec']
        g = df.groupby(key)
        c = fn(g['value']).reset_index()
        c = c.rename({'value': 'consensus'}, axis=1)
        # Add the consensus to the user data
        key = ['ticker', 'fpedats']
        dfu = self.open_data(data, ['ticker', self.col_fpedats])
        dfu['fpedats'] = dfu[self.col_fpedats]
        cons = dfu.merge(c[key+['consensus']], how='left', on=key)
        cons.index = dfu.index
        # Return the consensus
        return(cons.consensus.astype('float32'))

    def get_ptg(self, data, startdt, enddt, measure='PTG',
                horizon=12, n=5, stat='mean'):
        """
        Return analysts consensus price target from a given period for a
        given horizon.
        Note: No adjustment for splits is made.
        Arguments:
            data -- User provided dataset
                    Required columns: [ticker, startdt, enddt]
            startdt --  Beginning of the period used to compute the consensus.
            enddt --    End of the period used to compute the consensus.
            measure -- Measureto use (default 'PTG')
            horizon --  Defines the horizon to use (default=12 months)
            n --    Number of forecasts to use to compute the consensus
                    (default 5)
            stat -- Statistic to compute the consensus ('mean', 'median').
                    Default 'mean'.
        This function requires the following object attributes:
            self.ptgdet -- Data from the IBES 'ptgdet' file (price targets)
                           Required columns:   [ticker, horizon, value,
                                                amaskcd, anndats, anntims,
                                                measure, usfirm]
        """
        # Filter measure
        measures = ['PTG']
        if measure not in measures:
            raise Exception('Supported measures:', measures)
        det = self.open_data(self.ptg, ['measure'])
        kmeasure = det.measure == measure
        # Filter usfirm
        det = self.open_data(self.ptg, ['usfirm'])
        kusfirm = det.usfirm == self.usfirm
        # Filter horizon
        det = self.open_data(self.ptg, ['horizon'])
        khorizon = det.horizon == horizon
        # Create the final filter
        mask = kmeasure & kusfirm & khorizon
        cols = ['ticker', 'value', 'anndats', 'anntims', 'amaskcd']
        det = self.open_data(self.ptg, cols)
        det = det[mask].drop_duplicates()
        # Create a full timestamp for the forecasts
        det.anntims = pd.to_timedelta(det.anntims, unit='s')
        det['time'] = pd.to_datetime(det.anndats) + det.anntims
        # TODO: unadust the values with adjustment factor
        # Get the start and end date to compute
        # the consensus by (ticker, fpedats)
        key = ['ticker']
        dfu = data[key + [startdt, enddt]].drop_duplicates()
        # Keep the rows for which a consensus can be computed
        dfu = dfu.dropna()
        # Merge the datasets to compute the consensus
        cols_det = ['ticker', 'time', 'value']
        df = dfu.merge(det[cols_det], how='left', on=key)
        # Keep the analysts forecasts to consider
        df = df[(df.time >= pd.to_datetime(df[startdt])) &
                (df.time <= pd.to_datetime(df[enddt]))]
        # Keep the last n analysts forecasts
        df = df.sort_values(['ticker', startdt, enddt, 'time'])
        key = ['ticker', startdt, enddt]
        df = df.groupby(key).head(n+1)
        # Select the correct statistical function
        fn = getattr(pd.core.groupby.GroupBy, stat)
        # Compute the consensus
        g = df.groupby(key)
        c = fn(g['value']).reset_index()
        c = c.rename({'value': 'consensus'}, axis=1)
        # Add the consensus to the user data
        dfu = data[key]
        cons = dfu.merge(c[key+['consensus']], how='left', on=key)
        cons.index = dfu.index
        # Return the consensus
        return(cons.consensus.astype('float32'))

    def get_numanalys(self, data):
        """ Return the number of analysts. """
        key = ['ticker', 'fpedats']
        df = self._get_fields_det(key+['analys'])
        na = df.groupby(key).analys.nunique().reset_index(name='numanalys')
        # Merge to the user data
        dfu = self.open_data(data, ['ticker', self.col_fpedats])
        dfu['fpedats'] = dfu[self.col_fpedats]
        numan = dfu.merge(na, how='left', on=key)
        numan.index = dfu.index
        # Return the consensus
        return(numan.numanalys.astype('float32'))
