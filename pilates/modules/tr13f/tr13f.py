"""
Provides classes and functions to process Thomson Reuters data from WRDS.
"""

from pilates import wrds_module
import pandas as pd


class tr13f(wrds_module):

    def __init__(self, d):
        wrds_module.__init__(self, d)

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
            self.crsp.msf --  CRSP 'msf' file (monthly stock file)
            self.crsp.msenames -- CRSP 'msenames' file
        """
        # Check requirements
        self.d._check_id_column()
        self.d._check_date_column()

        ###############################################
        # Process the Holdings data (TR-13F S34type3) #
        ###############################################
        cols = ['fdate', 'mgrno', 'cusip', 'shares']
        hol = self.open_data(self.type3, cols)
        # Add permno information
        hol['permno'] = self.d.crsp.permno_from_cusip(hol)
        hol = hol[hol.permno.notna()]

        ########################################
        # Merge Manager data (TR-13f S34type1) #
        ########################################
        cols = ['rdate', 'mgrno', 'fdate']
        # key = ['rdate', 'mgrno']
        df = self.open_data(self.type1, cols)
        # Keep first vintage with holdings data for each (rdate, mgrno)
        df = df.groupby(['rdate', 'mgrno']).fdate.min().reset_index()
        df = df.drop_duplicates()
        # Merge the data
        m = pd.merge(hol, df)

        ###################################
        # Compute Institutional Ownership #
        ###################################
        # Setup the crsp module to work on monthly data
        user_crsp_freq = self.d.crsp.freq    # Save the current CRSP frequency
        user_col_date = self.d.col_date      # Save the current col_date
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
        self.d.crsp.freq = user_crsp_freq    # Set the CRSP frequency back
        self.d.col_date = user_col_date      # Set the col_date back

        ############################
        # Merge with the user data #
        ############################
        dfu = self.open_data(data, [self.d.col_id, self.d.col_date])
        if self.d.col_id == 'gvkey':
            dfu['permno'] = self.d.crsp.permno_from_gvkey(dfu)
        elif self.d.col_id != 'permno':
            raise Exception("Institutional Ownership using tr_13f data can "
                            "only be computed for data with identifiers 'gvkey' "
                            " or 'permno'.")
        # Merge the IO data (on permno, rdate)
        ## Correct columns types from the groupby on 'permno'
        io = self.d.crsp._correct_columns_types(io)
        ## Prepare the data
        dfu = dfu.sort_values(self.d.col_date)
        dfu = dfu.dropna()
        io = io.sort_values('rdate')
        ## IO data is quarterly so merge with quarterly tolerance
        dfin = pd.merge_asof(dfu, io,
                             left_on=self.d.col_date,
                             right_on='rdate',
                             by='permno',
                             tolerance=pd.Timedelta('90 day'),
                             direction='nearest')
        dfin.index = dfu.index
        return(dfin.io_frac)
