"""
Module to provide processing functions for Compustat Execucomp data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np


class execucomp(wrds_module):
    """ Class providing main processing methods for the Compustat Execucomp data.
    """

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_id = 'gvkey'
        self.col_date = 'year'
        #self.key = [self.col_id, self.col_date]

    def participantid_from_execid(self, data, match='11'):
        """ Returns ISS participant id from execucomp (gvkey, execid).
        Requires execid and gvkey.
        """
        # Get the linktable
        link = self.d.iss.get_link_iss_execucomp()
        # Count number of execid and participantid for each participantid and execid
        link['Nexecid'] = link.groupby(['gvkey', 'participantid']).execid.transform('count')
        link['Nparticipantid'] = link.groupby(['gvkey', 'execid']).participantid.transform('count')

        if match=='11':
            link = link[(link.Nexecid==1) & (link.Nparticipantid==1)]
        else:
            print('Execucomp: matches other than 1-to-1 not yet implemented')
            link = link[(link.Nexecid==1) & (link.Nparticipantid==1)]

        link = link[['gvkey', 'execid','participantid']]

        dfin = pd.merge(data, link, how='left', on=['gvkey', 'execid'])
        dfin.index = data.index
        return(dfin.participantid)

    def get_fields_anncomp(self, fields, data=None, role=None):
        """ Get the fields from table anncomp. """
        add_fields = []
        ceo = False
        cfo = False
        if role in ['CEO', 'ceo']:
            # Select observations for CEOs
            add_fields = [f for f in ['ceoann', 'titleann', 'execdir'] if f not in fields]
            ceo = True
            cfo = False
        elif role in ['CFO', 'cfo']:
            # Select observations for CFOs
            add_fields = [f for f in ['cfoann', 'titleann'] if f not in fields]
            ceo = False
            cfo = True

        # Open execucomp data
        key = [self.col_id, self.col_date, 'execid']
        df = self.open_data(self.anncomp, key+fields+add_fields).drop_duplicates()
        # Drop duplicates (pick the first of the duplicate)
        dupi = df[key].duplicated()
        df = df[~dupi]


        key = [self.col_id, self.col_date]
        if ceo:
            cond1 = df.ceoann.str.lower().str.contains('ceo')
            #cond2 = df.titleann.str.lower().str.contains('ceo')
            #cond3 = df.titleann.str.lower().str.contains('chief executive officer')
            #dfceo = df[cond1 | cond2 | cond3]
            dfceo = df[cond1]
            # Remove duplicates
            dupi = dfceo[key].duplicated(keep=False)
            # 1) Keep non-duplicates
            dfceo1 =  dfceo[~dupi]
            # 2) Disentangle duplicates
            dup1 = dfceo[dupi]
            # Remove
            # Remove non-execdir
            tmp2 = dup1[dup1.execdir==1]
            dupi2 = tmp2[key].duplicated(keep=False)
            dfceo2 = tmp2[~dupi2]
            dup2 = tmp2[dupi2]
            # Keep the ones with CEO title
            dup2['titleceo'] = 0
            dup2.loc[dup2.titleann.str.lower().str.contains('ceo'), 'titleceo'] = 1
            dup2.loc[dup2.titleann.str.lower().str.contains('chief executive officer'), 'titleceo'] = 1
            tmp3 = dup2[dup2.titleceo==1]
            dupi3 = tmp3[key].duplicated(keep=False)
            dfceo3 = tmp3[~dupi3].drop(columns=['titleceo'])
            dup3 = tmp3[dupi3].drop(columns=['titleceo'])
            # Concatenate all non duplicates
            df = pd.concat([dfceo1, dfceo2, dfceo3])[key+fields]
        elif cfo:
            # Multiple conditions to select CFOs
            cond1 = df.cfoann.str.lower() == 'cfo'
            cond2 = df.titleann.str.lower().str.contains('cfo')
            cond3 = df.titleann.str.lower().str.contains('chief financial officer')
            cond4 = df.titleann.str.lower().str.contains('treasurer')
            cond5 = df.titleann.str.lower().str.contains('controller')
            cond6 = df.titleann.str.lower().str.contains('finance')
            cond7 = df.titleann.str.lower().str.contains('vice president-finance')
            dfcfo = df[cond1 | cond2 | cond3 | cond4 | cond5 | cond6 | cond7]
            # Remove duplicates
            dupi = dfcfo[key].duplicated(keep=False)
            # 1) Keep non-duplicates
            dfcfo1 =  dfcfo[~dupi]
            # 2) Disentangle duplicates
            dup1 = dfcfo[dupi]
            # Keep the ones with 'cfo' or 'chief financial officer'
            dup1['titlecfo'] = 0
            dup1.loc[dup1.titleann.str.lower().str.contains('cfo'), 'titlecfo'] = 1
            dup1.loc[dup1.titleann.str.lower().str.contains('chief financial officer'), 'titlecfo'] = 1
            tmp2 = dup1[dup1.titlecfo==1]
            dupi2 = tmp2[key].duplicated(keep=False)
            dfcfo2 = tmp2[~dupi2].drop(columns=['titlecfo'])
            dup2 = tmp2[dupi2].drop(columns=['titlecfo'])
            # Keep the ones with cfoann='CFO'
            tmp3 = dup2[dup2.cfoann=='CFO']
            dupi3 = tmp3[key].duplicated(keep=False)
            dfcfo3 = tmp3[~dupi3]
            dup3 = tmp3[dupi3]
            # Concatenate all non duplicates
            df = pd.concat([dfcfo1, dfcfo2, dfcfo3])[key+fields]

        if data is not None:
            # Merge to user's data and returns only the requested fields
            key = [self.col_id, self.col_date, 'execid']
            dfin = pd.merge(data, df, how='left',
                            left_on=[self.col_id, self.d.col_date, 'execid'],
                            right_on=key)
            dfin.index = data.index
            return(dfin[fields])
        else:
            return(df)

    def tenure(self, data):
        """ Compute the executive tenure. """
        None

    def experience_firm(self, data):
        """ Compute the executive experience in the firm (in years). """
        key = [self.col_id, self.col_date, 'execid']
        fields = ['joined_co']
        df = self.open_data(self.anncomp, key+fields).drop_duplicates().sort_values(['execid', 'gvkey', 'year'])
        # Compute using joined_co field
        df['exp_jc'] = df.year - df.joined_co.dt.year
        # Compute using observed data
        df['exp_rol'] = df.groupby(['execid','gvkey']).cumcount()
        # Combine both
        df['exp'] = df[['exp_jc', 'exp_rol']].max(axis=1).astype('Int32')
        # Add to user data
        dfin = pd.merge(data, df, how='left',
                        left_on=[self.col_id, self.d.col_date, 'execid'],
                        right_on=key)
        dfin.index = data.index
        return(dfin['exp'])

    def experience_industry(self, data, sic=None, naics=None):
        """ Compute the executive experience in the industry (in years).

        Arguments:
            sic     Specify the number of SIC digits to consider ('2d', '3d', '4d')
                    if use SIC classification.
            naics   Specify the number of SIC digits to consider ('1d', '2d', '3d', '4d', '5d, '6d')
                    if use NAICS classification.
        """
        key = [self.col_id, self.col_date, 'execid']
        if sic is None and naics is None:
            raise Exception('Execucomp: Please select an industry classification code with the number of',
                             'digits to use.')
        elif sic is not None and naics is not None:
            raise Exception('Execucomp: Please select wither SIC or NAICS classification.')
        elif sic is not None:
            fields = ['joined_co', 'sic']
            colind = 'sic'
        elif naics is not None:
            fields = ['joined_co', 'naics']
            colind = 'naics'
        # Add industry to user data
        dfu = data.copy()
        dfu[colind] = self.get_fields_anncomp([colind], data=dfu)
            # Add industry to user data
        # Fetch anncomp data
        df = self.open_data(self.anncomp, key+fields).drop_duplicates()
        # Remove observations without industry information
        df = df.dropna(subset=[colind])
        # Create the industry code
        if sic is not None:
            if sic=='2d':
                df['ind'] = (df[colind] / 100).astype(int)
                dfu.loc[dfu.sic.notnull(), 'ind'] = (dfu.loc[dfu.sic.notnull(), colind] / 100).astype(int)
            if sic=='3d':
                df['ind'] = (df[colind] / 10).astype(int)
                dfu.loc[dfu.sic.notnull(), 'ind'] = (dfu.loc[dfu.sic.notnull(), colind] / 10).astype(int)
            if sic=='4d':
                df['ind'] = df['sic']
                dfu['ind'] = dfu['sic']
        if naics is not None:
            N = int(naics[0])
            df['ind'] = df[colind].astype(str).str[:N].astype(int)
            dfu.loc[dfu.naics.notnull(), 'ind'] = dfu.loc[dfu.naics.notnull(), colind].astype(str).str[:N].astype(int)
        # Count number of years in the data
        dfin = df.sort_values(['execid', 'ind', 'year'])[['execid', 'ind', 'year']].drop_duplicates()
        dfin['exp_rol'] = dfin.groupby(['execid', 'ind']).cumcount()
        #df = pd.merge(df, dfin, how='left')
        # Correct using 'joined_co' when possible
        #df['exp_jc'] = df.year - df.joined_co.dt.year
        # Find earliest date at which the data is available in a given industry
        jc = df[df.joined_co.notnull()][['execid', 'ind', 'gvkey', 'year', 'joined_co']]
        jc = jc.sort_values(['execid', 'ind', 'year', 'gvkey'])
        jc['min_year'] = jc.groupby(['execid', 'ind'])['year'].transform('min')
        jcm = jc[jc.year==jc.min_year]
        # If duplicates, use the one with oldest joined_co
        tmp = jcm[['execid', 'ind', 'year', 'min_year']].drop_duplicates()
        tmp['joined_co'] = jcm.groupby(['execid', 'ind', 'year', 'min_year']).joined_co.transform('min')
        jcm = tmp
        jcm['expi'] = jcm.year - jcm.joined_co.dt.year
        jcm = pd.merge(jcm, dfin, how='left')
        jcm.rename(columns={'exp_rol': 'exp_rol_min_year'}, inplace=True)
        # Add info to final data
        jcm2 = jcm[['execid', 'ind', 'expi', 'min_year', 'exp_rol_min_year']].drop_duplicates()
        dfin2 = pd.merge(dfin, jcm2, how='left')
        # Correct the number of years
        dfin2.loc[dfin2.expi.notnull(), 'exp_jc'] = dfin2.expi - dfin2.exp_rol_min_year + dfin2.exp_rol
        # Combine both
        dfin2['exp'] = dfin2[['exp_jc', 'exp_rol']].max(axis=1).astype('Int32')
        df = dfin2[['execid', 'ind', 'year', 'exp']]

        # Add to user data
        key = ['execid', 'ind', 'year']
        dfin = pd.merge(dfu, df, how='left',
                        left_on=['execid', 'ind', self.d.col_date],
                        right_on=key)
        dfin.index = data.index
        return(dfin['exp'])
