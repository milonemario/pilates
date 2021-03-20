"""
Provides classes and functions to process WRDS SEC data.
"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import datetime


class sec13f(wrds_module):

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values

    def get_io_perc(self, data):
        """ Returns the fraction of Institutional Ownership from 13F holdings.
        This script is based on the WRDS code from the
        'Research Note Regarding Thomson-Reuters Ownership Data Issues'
        from May 2017.
        Arguments:
            data -- User provided data
                    Required columns:   [permno, 'col_date']
                                        or [gvkey, 'col_date']
        This function requires the following object attributes:
            self.holdings --    Data from the 'wrds_13f_holdings' file
        """
        ##########################
        # Clean 13F Summary data #
        ##########################
        # Keep one filing per quarter (select the first 'correct' filing)
        # Either only one filing: Keep the one
        # If multiple filings: Keep the most recent restatement within one
        # month of the original filing.
        key = ['coname', 'rdate']
        cols = ['cik', 'fname', 'fdate', 'rdate', 'coname', 'reporttype',
                'amendmenttype', 'confdeniedexpired', 'tableentrytotal']
        su = self.open_data(self.summary, cols).drop_duplicates()
        su = su[su.rdate >= datetime.date(2013, 6, 30)]
        su['first_fdate'] = su.groupby(key).fdate.transform('min')
        # Only consider restatements within one month after the first filing
        su = su[su.fdate - su.first_fdate < np.timedelta64(31, 'D')]
        su['n13f'] = su.groupby(key).fdate.transform('count')
        # Select the correct filing:
        su1 = su[su.n13f == 1]
        sun = su[su.n13f > 1]
        # Discard the last filing when there are more and keep
        # - The non-denied/non-expired confidential treatment
        # - The restatements only
        sun = sun[(sun.fdate != sun.first_fdate) &
                  (sun.confdeniedexpired != 'true') &
                  (sun.amendmenttype == 'RESTATEMENT')]
        # Keep the most recent restatement
        sun['last_fdate'] = sun.groupby(key).fdate.transform('max')
        sun = sun[sun.fdate == sun.last_fdate]
        # Concatenate all cleaned filings
        suf = pd.concat([su1, sun])
        suf = suf[['cik', 'fname', 'fdate', 'rdate', 'coname']]

        ###########################
        # Merge the Holdings data #
        ###########################
        cols = ['cusip', 'fname', 'sshprnamt']
        hol = self.open_data(self.holdings, cols)
        m = suf.merge(hol, how='left', on='fname')
        m = m.rename({'sshprnamt': 'shares'}, axis=1)
        # Keep data after Q2-2013 when it became required
        m['cusip'] = m.cusip.str.slice(start=0, stop=8)
        m['permno'] = self.d.crsp.permno_from_cusip(m)
        # Setup the crsp module to work on monthly data
        crsp_freq = self.d.crsp.freq    # Save the current CRSP frequency
        col_date = self.d.col_date      # Save the current col_date
        self.d.crsp.set_frequency('Monthly')
        self.d.set_date_column('fdate')
        # Adjust the shares held
        m['s_adj'] = self.d.crsp._adjust_shares(m, col_shares='shares')

        # ##################
        # # Clean the data #
        # ##################
        # # Blackrock Fix (Ben-David, Franzoni, Moussawi and Sedunov (2016))
        # br_ciks = ['0001003283', '0001006249', '0001085635', '0001086364',
        #            '0001305227', '0001364742']
        # m.loc[m.cik.isin(br_ciks), 'cik'] = '0000913414'
        # # Compute the aggregate holding per institution for each permno
        # key = ['cik', 'rdate', 'permno']
        # ma = m.groupby(key).s_adj.sum().reset_index()
        # # Add the mgrno to the data
        # # 1 - Clean the mgrno
        # cr = ma[['cik', 'rdate']].drop_duplicates()
        # # Add the link to mgrno (from TR-S34) using WRDS link file
        # link = self.open_data(self.link, ['cik', 'mgrno',
        #                                     'matchrate', 'flag'])
        # cr = cr.merge(link, how='left', on='cik')
        # # # Flag the mgrno that are present on TR-S34 after 2013-06-30
        # # s34 = self.open_data(self.d.tr13f.type1, ['mgrno', 'fdate'])
        # # s34 = s34[s34.fdate.dt.date >= datetime.date(2013, 6, 30)]
        # # s34mgrnos = s34.mgrno.unique()
        # # cr['ins34'] = False
        # # cr.loc[cr.mgrno.isin(s34mgrnos), 'ins34'] = True
        # # If multiple cik-rdate, keep
        # # - the highest flag or
        # # - the highest match or
        # # - the mgrno in the s34 file after Q2-2013
        # cr['hflag'] = cr.groupby(['cik', 'rdate']).flag.transform('max')
        # cr = cr[(cr.flag == cr.hflag) | (cr.mgrno.isna())]
        # cr['hmr'] = cr.groupby(['cik', 'rdate']).matchrate.transform('max')
        # cr = cr[(cr.matchrate == cr.hmr) | (cr.mgrno.isna())]
        # # If still duplicates, just leave them. TODO
        # # If no mgrno, use -cik
        # cr.loc[cr.mgrno.isna(), 'mgrno'] = (-cr.cik.astype(int)).astype(str)
        # # Merge it to the data
        # mam = ma.merge(cr[['cik', 'rdate', 'mgrno']],
        #                how='left', on=['cik', 'rdate'])

        ###################################
        # Compute Institutional Ownership #
        ###################################
        # There might be multiple cusips for one permno: Consider them
        # all (group the holdings by permno).
        # Compute the total IO shares
        io = m.groupby(['permno',
                        'rdate']).s_adj.sum().reset_index(name='io')
        # Get the total number of shares from CRSP
        self.d.set_date_column('rdate')
        io['tso'] = self.d.crsp._tso(io)
        # Compute the IO fraction
        io['io_frac'] = io.io / io.tso
        self.d.crsp.freq = crsp_freq    # Set the CRSP frequency back
        self.d.col_date = col_date      # Set the col_date back

        ############################
        # Merge with the user data #
        ############################
        data_cols = self.d.get_fields_names(data)
        if 'permno' in data_cols:
            dfu = self.open_data(data, ['permno', self.d.col_date])
        elif 'gvkey' in data_cols:
            dfu = self.open_data(data, ['gvkey', self.d.col_date])
            dfu['permno'] = self.d.crsp.permno_from_gvkey(dfu)
        else:
            raise Exception('get_io_perc only accepts permno or gvkey ' +
                            ' as identifiers')
        # index = dfu.index
        # dfu = dfu.dropna()
        if self.d.freq in ['Q', 'M']:
            # Merge on year and month
            dt = pd.to_datetime(dfu[self.d.col_date]).dt
            dfu['year'] = dt.year
            dfu['month'] = dt.month
            dt = pd.to_datetime(io.rdate).dt
            io['year'] = dt.year
            io['month'] = dt.month
            key = ['permno', 'year', 'month']
        elif self.d.freq in ['A']:
            # Keep the latest quarter of the year
            dt = pd.to_datetime(m.fdate).dt
            m['year'] = dt.year
            None
        elif self.d.freq in ['D']:
            # Merge on the day
            None
        dfin = dfu.merge(io[key+['io_frac']], how='left', on=key)
        dfin.index = dfu.index
        return(dfin.io_frac.astype('float32'))
