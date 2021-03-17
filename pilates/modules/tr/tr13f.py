"""
Provides classes and functions to process Thomson Reuters data.
"""

from pilates import data_module
import pandas as pd


class tr13f(wrds_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        # Initialize values

    def get_io_perc(self, data):
        """ Returns the fraction of Institutional Ownership from the S34 data.
        This function is based on the following WRDS SAS code:
            Summary :   Calculate Institutional Ownership, Concentration,
                        and Breadth Ratios
            Date    :   May 18, 2009
            Author  :   Luis Palacios, Rabih Moussawi, and Denys Glushkov
        Arguments:
            data -- User provided data
                    Required columns:   [permno, 'col_date']
                                        or [gvkey, 'col_date']
        This function requires the following object attributes:
            self.type1 --   Data from the TR s34 's34type1' file
                            (Manager)
            self.type3 --   Data from the TR s34 's34type3' file
                            (Stock Holdings)
            self.d.crsp.msf --  CRSP 'msf' file (monthly stock file)
            self.d.crsp.msenames -- CRSP 'msenames' file
        """
        ###############################################
        # Process the Holdings data (TR-13F S34type3) #
        ###############################################
        cols = ['fdate', 'mgrno', 'cusip', 'shares']
        hol = self.d.open_data(self.type3, cols)
        # Add permno information
        hol['permno'] = self.d.crsp.permno_from_cusip(hol)
        hol = hol[hol.permno.notna()]

        ########################################
        # Merge Manager data (TR-13f S34type1) #
        ########################################
        cols = ['rdate', 'mgrno', 'fdate']
        # key = ['rdate', 'mgrno']
        df = self.d.open_data(self.type1, cols)
        # Keep first vintage with holdings data for each (rdate, mgrno)
        df = df.groupby(['rdate', 'mgrno']).fdate.min().reset_index()
        df = df.drop_duplicates()
        # Merge the data
        m = pd.merge(hol, df)

        ###################################
        # Compute Institutional Ownership #
        ###################################
        # Setup the crsp module to work on monthly data
        crsp_freq = self.d.crsp.freq    # Save the current CRSP frequency
        col_date = self.d.col_date      # Save the current col_date
        self.d.crsp.set_frequency('Monthly')
        self.d.set_date_column('fdate')
        # Adjust the shares held
        m['s_adj'] = self.d.crsp._adjust_shares(m, col_shares='shares')
        # There might be multiple cusips for one permno: Consider them
        # all (group the holdings by permno).
        # Compute the total IO shares
        io = m.groupby(['permno', 'rdate']).s_adj.sum().reset_index(name='io')
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
            dfu = self.d.open_data(data, ['permno', self.d.col_date])
        elif 'gvkey' in data_cols:
            dfu = self.d.open_data(data, ['gvkey', self.d.col_date])
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
