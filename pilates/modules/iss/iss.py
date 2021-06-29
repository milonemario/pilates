"""
Module to provide processing functions for ISS data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import os, zipfile, re
from codecs import open
from rapidfuzz import fuzz


class iss(wrds_module):
    """ Class providing main processing methods for the ISS data.

    One instance of this classs is automatically created and accessible from
    any data object instance.

    Args:
        d (pilates.data): Instance of pilates.data object.

    """

    def __init__(self, d):
        wrds_module.__init__(self, d)

    def cik_from_gvkey(self, data):
        """ Returns ISS cik from Compustat gvkey.
        """
        None

    def gvkey_from_cik(self, data):
        """ Returns COMPUSTAT gvkey from ISS cik.
        """
        df_iss = self.open_data(self.companyfy, columns=['cik','companyname','ticker','cusip'])
        df_comp = self.d.comp.open_data(self.d.comp.names, columns=['gvkey','conm','cik','tic','cusip'])

        # Clean ISS companies names
        for c in [',','.']:
            df_iss['companyname'] = df_iss.companyname.str.replace(c,'',regex=False)
        df_iss['companyname'] = df_iss.companyname.str.upper()
        df_iss = df_iss.drop_duplicates()

        df = df_iss

        #############
        # Using CIK #
        #############
        df = pd.merge(df, df_comp[['cik', 'gvkey']], how='left')

        # Only 157 missing CIKs, so we stop here for now.

        ######################
        # Merge to user data #
        ######################
        df_gvkey = df[['cik','gvkey']].drop_duplicates().dropna()

        dfin = pd.merge(data['cik'], df_gvkey, how='left')
        # Return the gvkeys
        dfin.index = data.index
        return dfin.gvkey

    def get_link_iss_execucomp(self):
        ####################################################
        # Create a general match between Execucomp and ISS #
        ####################################################
        fields_iss = ['cik', 'participantid', 'fullname', 'lastname', 'middlename', 'firstname']
        fields_exe = ['gvkey', 'execid', 'exec_fullname', 'exec_lname', 'exec_mname', 'exec_fname']

        dfiss = self.open_data(self.participantfy, fields_iss)
        dfexe = self.d.execucomp.open_data(self.d.execucomp.anncomp, fields_exe)

        dfiss = dfiss.drop_duplicates()
        dfexe = dfexe.drop_duplicates()

        # Add gvkey information
        dfiss['gvkey'] = self.gvkey_from_cik(dfiss)

        dfexe['lastname'] = dfexe.exec_lname
        dfexe['middlename'] = dfexe.exec_mname
        dfexe['firstname'] = dfexe.exec_fname
        dfexe['fullname'] = dfexe.exec_fullname

        ##### Name cleanup #####
        # Remove '.', ',' and all cap
        for col in ['lastname', 'middlename', 'firstname', 'fullname']:
            for char in [',', '.']:
                dfiss[col] = dfiss[col].str.replace(char, '', regex=False)
                dfexe[col] = dfexe[col].str.replace(char, '', regex=False)
            dfiss[col] = dfiss[col].str.upper()
            dfexe[col] = dfexe[col].str.upper()

        ##### Merge #####
        ### Start from most strict to more lax and keep 1-1 matches
        dfkeeps = []
        ## For same firm ##
        # For same (lastname, middlename, firstname)
        # or same (lastname, firstname), allow for 1-N and N-1 matches
        keys =  [['lastname', 'middlename', 'firstname'], ['lastname', 'firstname']]
        for key in keys:
            df = pd.merge(dfiss[key+['gvkey', 'participantid']], dfexe[key+['gvkey', 'execid']])
            # Keep 1-1 matches
            tmp = df[['gvkey', 'participantid', 'execid']].drop_duplicates()
            tmp['Nexecid'] = tmp.groupby(['gvkey', 'participantid']).execid.transform('count')
            tmp['Nparticipantid'] = tmp.groupby(['gvkey', 'execid']).participantid.transform('count')
            dfk = tmp[(tmp.Nparticipantid==1) | (tmp.Nexecid==1)]
            dfkeeps += [dfk]
            keep_iss = dfk.participantid.unique()
            keep_exe = dfk.execid.unique()
            dfiss = dfiss[~dfiss.participantid.isin(keep_iss)]
            dfexe = dfexe[~dfexe.execid.isin(keep_exe)]

        # Otherwise allow only for 1-1 matches
        keys =  [['lastname'], ['fullname']]
        for key in keys:
            df = pd.merge(dfiss[key+['gvkey', 'participantid']], dfexe[key+['gvkey', 'execid']])
            # Keep 1-1 matches
            tmp = df[['gvkey', 'participantid', 'execid']].drop_duplicates()
            tmp['Nexecid'] = tmp.groupby(['gvkey', 'participantid']).execid.transform('count')
            tmp['Nparticipantid'] = tmp.groupby(['gvkey', 'execid']).participantid.transform('count')
            dfk = tmp[(tmp.Nparticipantid==1) & (tmp.Nexecid==1)]
            dfkeeps += [dfk]
            keep_iss = dfk.participantid.unique()
            keep_exe = dfk.execid.unique()
            dfiss = dfiss[~dfiss.participantid.isin(keep_iss)]
            dfexe = dfexe[~dfexe.execid.isin(keep_exe)]

        ## Regardless of firm
        # Only allow 1-1 matches
        keys =  [['lastname', 'middlename', 'firstname'],
                ['fullname']]
        for key in keys:
            df = pd.merge(dfiss[key+['participantid']], dfexe[key+['execid']])
            # Keep 1-1 matches
            tmp = df[['participantid', 'execid']].drop_duplicates()
            tmp['Nexecid'] = tmp.groupby(['participantid']).execid.transform('count')
            tmp['Nparticipantid'] = tmp.groupby(['execid']).participantid.transform('count')
            dfk = tmp[(tmp.Nparticipantid==1) & (tmp.Nexecid==1)]
            dfkeeps += [dfk]
            keep_iss = dfk.participantid.unique()
            keep_exe = dfk.execid.unique()
            dfiss = dfiss[~dfiss.participantid.isin(keep_iss)]
            dfexe = dfexe[~dfexe.execid.isin(keep_exe)]

        ### Fuzzy matching
        ## For same firm
        key = ['gvkey', 'lastname', 'firstname', 'fullname']
        df = pd.merge(dfiss[key+['participantid']], dfexe[key+['execid']], on='gvkey').dropna()

        df['lastname_ratio'] = df.apply(lambda row: fuzz.partial_ratio(row.lastname_x, row.lastname_y), axis=1)
        df['firstname_ratio'] = df.apply(lambda row: fuzz.partial_ratio(row.firstname_x, row.firstname_y), axis=1)
        df['fullname_ratio'] = df.apply(lambda row: fuzz.partial_ratio(row.fullname_x, row.fullname_y), axis=1)

        # On first and last name - allow for 1-N and N-1 matches
        tmp = df[(df.lastname_ratio==100) & (df.firstname_ratio==100)][['gvkey', 'participantid', 'execid']].drop_duplicates()
        tmp['Nexecid'] = tmp.groupby(['gvkey', 'participantid']).execid.transform('count')
        tmp['Nparticipantid'] = tmp.groupby(['gvkey', 'execid']).participantid.transform('count')
        dfk = tmp[(tmp.Nparticipantid==1) | (tmp.Nexecid==1)]
        dfkeeps += [dfk]
        keep_iss = dfk.participantid.unique()
        keep_exe = dfk.execid.unique()
        dfiss = dfiss[~dfiss.participantid.isin(keep_iss)]
        dfexe = dfexe[~dfexe.execid.isin(keep_exe)]

        df = df[~df.participantid.isin(keep_iss) | ~df.execid.isin(keep_exe)]

        # On full name - allow 1-N and M-1 matches
        tmp = df[(df.fullname_ratio==100)][['gvkey', 'participantid', 'execid']].drop_duplicates()
        tmp['Nexecid'] = tmp.groupby(['gvkey', 'participantid']).execid.transform('count')
        tmp['Nparticipantid'] = tmp.groupby(['gvkey', 'execid']).participantid.transform('count')
        dfk = tmp[(tmp.Nparticipantid==1) | (tmp.Nexecid==1)]
        dfkeeps += [dfk]
        keep_iss = dfk.participantid.unique()
        keep_exe = dfk.execid.unique()
        dfiss = dfiss[~dfiss.participantid.isin(keep_iss)]
        dfexe = dfexe[~dfexe.execid.isin(keep_exe)]

        df = df[~df.participantid.isin(keep_iss) | ~df.execid.isin(keep_exe)]

        link = pd.concat(dfkeeps)[['gvkey', 'participantid', 'execid']].drop_duplicates()
        return link

    def execid_from_participantid(self, data, match='11'):
        """ Returns Execucomp execid from iss (gvkey, participantid).
        Requires participantid and either gvkey or cik.
        """
        # Get the linktable
        link = self.d.iss.get_link_iss_execucomp()
        # Count number of execid and participantid for each participantid and execid
        link['Nexecid'] = link.groupby(['gvkey', 'participantid']).execid.transform('count')
        link['Nparticipantid'] = link.groupby(['gvkey', 'execid']).participantid.transform('count')

        if match=='11':
            link = link[(link.Nexecid==1) & (link.Nparticipantid==1)]
        else:
            print('ISS : matches other than 1-to-1 not yet implemented')
            link = link[(link.Nexecid==1) & (link.Nparticipantid==1)]

        # Add gvkey information if not present
        dfu = data.copy()
        if 'gvkey' not in data.columns:
            dfu['gvkey'] = self.gvkey_from_cik(dfu)

        link = link[['gvkey', 'participantid', 'execid']]

        dfin = pd.merge(dfu, link, how='left', on=['gvkey', 'participantid'])
        dfin.index = dfu.index
        return(dfin.execid)


        ###################################
        # Merge execid to the user's data #
        ###################################
        # If gvkey is not present, add it

        # Add execid
