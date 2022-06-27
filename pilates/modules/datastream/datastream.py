"""
Module to provide processing functions for Compustat data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import product
from functools import partial


class datastream(wrds_module):
    """ Class providing main processing methods for the Worldscope data.
    """

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_id = 'infocode'
        self.col_date = 'marketdate'
        self.key = [self.col_id, self.col_date]

    def _get_holidays(self):
        """ Returns trading holidays.
        Follows code from
        "Research Guide for Datastream and Worldscope at Wharton Research Data Services"
        by Rui Dai and Qingyi (Freda) Drechsler
        """
        # Open the required tables
        sdif = self.open_data(self.SDExchInfo_v, columns=['exchcode', 'isoctry', 'holtype', 'cntryname'])
        sdd = self.open_data(self.SDDates_v)
        sdi = self.open_data(self.SDInfo_v)
        xref = self.open_data(self.DS2XRef)
        xref = xref[xref.type_ == 1017]   # Trading Holiday Mapping
        xref.desc_ = xref.desc_.astype('Int32')  # Corresponds to exchcode
        xref.code = xref.code.astype(int).astype('Int32')  # Corresponds to exchintcode
        exch = self.open_data(self.DS2Exchange, columns=['exchintcode'])
        # Merge
        df = sdif.merge(sdd)
        df = df.merge(sdi)
        df = df.merge(xref, left_on='exchcode', right_on='desc_', suffixes=('_sdd', '_xref'))
        df = df.merge(exch, left_on='code_xref', right_on='exchintcode')
        # Select, rename and return
        df = df[['exchintcode', 'cntryname', 'isoctry', 'holtype', 'date_', 'name']]
        df.rename({'date_': 'holidays', 'name': 'holiday_name'}, axis=1, inplace=True)
        return(df)

    def _value_for_data(self, var, data, ascending, useall, tolerance):
        """" Add values to the users data and return the values.
        Arguments:
            var --  Series containing the values,
                    indexed by (self.col_id, self.col_date)
            data -- User data
                    Columns: [self.col_id, pilates.col_date]
            ascending -- Boolean
            useall --  If True, use the value of the last
                        available date (if ascending is True) or the
                        value of the next available date
                        (if ascending is False).
            tolerance -- maximum time period to look for a closest match on the
                         date.
        """
        key = self.key
        var.name = 'var'
        values = var.reset_index()
        # Make sure the types are correct
        values = self._correct_columns_types(values)
        # Open user data
        cols_data = [self.col_id, self.d.col_date]
        dfu = self.open_data(data, cols_data)
        # Prepare the dataframes for merging
        dfu = dfu.sort_values(self.d.col_date)
        dfu = dfu.dropna()
        values.sort_values(self.col_date, inplace=True)
        if useall:
            # Merge on col_id and on closest date
            # Use the last or next trading day if requested
            if ascending:
                direction = 'backward'
            else:
                direction = 'forward'
            dfin = pd.merge_asof(dfu, values,
                                 left_on=self.d.col_date,
                                 right_on=self.col_date,
                                 by=self.col_id,
                                 tolerance=tolerance,
                                 direction=direction)
        else:
            dfin = dfu.merge(values, how='left', left_on=cols_data, right_on=self.key)
        dfin.index = dfu.index
        return(dfin['var'].astype('float32'))

    def _compounded_return_for_id(self, ids, hol, data, window, min_periods, logreturn,
                                  ascending, useall, tolerance):
        """ Compute compounded return for a list of (or one) given Datastream security(ies).
        Use approach dexcribed in
        "Research Guide for Datastream and Worldscope at Wharton Research Data Services"
        by Rui Dai and Qingyi (Freda) Drechsler
        Apply the following filters:
            - Exclude trading holidays.
            - Exclude problematic returns (see WRDS paper above)
            - Drop returns after delisting (after only null or zero returns available).
        """
        if type(ids) == int:
            filters = [[(self.col_id, '=', ids)]]
            dfu = data[data[self.col_id]==ids]
        elif type(ids) == list:
            filters = [[(self.col_id, 'in', ids)]]
            dfu = data[data[self.col_id].isin(ids)]

        sf = self.open_data(self.DS2PrimQtRI, filters=filters)
        # Exclude trading holidays
        sf = sf.merge(hol, how='left',
                     left_on=['marketdate', 'exchintcode'],
                     right_on=['holidays', 'exchintcode'])
        sf = sf[sf.holidays.isna()]
        sf.drop('holidays', axis=1, inplace=True)
        # Compute return from RI numbers
        sf['ri_lag'] = sf.groupby(self.col_id).ri.shift(1)
        sf['marketdate_lag'] = sf.groupby(self.col_id).marketdate.shift(1)
        sf['ret'] = (sf.ri / sf.ri_lag) - 1
        # Deal with returns problems
        # Follow Griffin et al. (2010) and Ince and Porter (2006)
        # as mention in WRDS paper.
        sf['ret_lag'] = sf.groupby(self.col_id).ret.shift(1)
        sf['ret_lead'] = sf.groupby(self.col_id).ret.shift(-1)
        sf['marketdate_delta'] = (sf.marketdate - sf.marketdate_lag).dt.days
        sf['compound_lag'] = (1+sf.ret)*(1+sf.ret_lag) - 1
        sf['compound_lead'] = (1+sf.ret)*(1+sf.ret_lead) - 1
        adjacent = (sf.marketdate_delta > 0) & (sf.marketdate_delta <= 7)
        h_ret_lag = (sf.ret>1) | (sf.ret_lag>1)
        h_ret_lead = (sf.ret>1) | (sf.ret_lead>1)
        abn_rev = adjacent & (h_ret_lag | h_ret_lead) & (sf.compound_lag<.2)
        sf.drop(['ri', 'ri_lag', 'marketdate_lag', 'ret_lag', 'ret_lead',
                 'marketdate_delta', 'compound_lag', 'compound_lead'],
                 axis=1, inplace=True)
        sf = sf[~abn_rev]
        # Deal with delisting
        # Get last non-null and non-missing return per security
        sf_fin = sf[~sf.ret.isna() & sf.ret!=0]
        last_list = sf_fin.groupby(self.col_id)[self.col_date].max().reset_index()
        last_list.rename({self.col_date: 'last_date'}, axis=1, inplace=True)
        sf = sf.merge(last_list, how='left')
        sf = sf[sf[self.col_date] <= sf.last_date]
        sf.drop('last_date', axis=1, inplace=True)

        # Compute the compounded returns
        with np.errstate(divide='ignore'):
            sf['ln1ret'] = np.log(1 + sf.ret)
        if window!=1:
            sf.set_index(self.col_date, inplace=True)
            sf.sort_index(inplace=True)
            sumln = sf.groupby(self.col_id).rolling(window, min_periods=min_periods).ln1ret.sum()
        else:
            sf.set_index(self.key, inplace=True)
            sumln = sf.ln1ret
        if logreturn:
            cret = sumln
        else:
            cret = np.exp(sumln) - 1
        # Add values to user data
        return(self._value_for_data(cret, dfu, ascending, useall, tolerance))

    def compounded_return(self, data, nperiods=1, caldays=None, min_periods=None,
                          logreturn=False, useall=True, tolerance=pd.Timedelta('6 day'),
                          chunk_max_ids=None, verbose=False, parallel=False):
        """ Compute compounded returns
        Use approach described in
        "Research Guide for Datastream and Worldscope at Wharton Research Data Services"
        by Rui Dai and Qingyi (Freda) Drechsler
        """
        # Get the window and sort
        window, ascending = self._get_window_sort(nperiods, caldays, min_periods)
        # Check arguments
        if nperiods==0:
            raise Exception("nperiods must be different from 0.")
        # Open the necessary data
        # Use only the IDs present in the user's data
        ids = data[self.col_id].dropna().drop_duplicates().to_list()
        # Get trading holidays
        hol = self._get_holidays()[['exchintcode', 'holidays']]
        # Use only the necessary columns from user data
        dfu = data[[self.col_id, self.d.col_date]]

        if chunk_max_ids is None:
            res_all = self._compounded_return_for_id(ids, hol, dfu, window, min_periods,
                                                     logreturn, ascending, useall, tolerance)
        else:
            # Process the data by chunk_max_ids ids at a time
            # Split ids by chunks
            chunks_ids = [ids[i:i+chunk_max_ids] for i in range(0,len(ids), chunk_max_ids)]
            if parallel:
                fn = partial(self._compounded_return_for_id, hol=hol, data=dfu,
                             window=window, min_periods=min_periods, logreturn=logreturn,
                             ascending=ascending, useall=useall, tolerance=tolerance)
                pool = mp.Pool()
                res_chunks = pool.map(fn, chunks_ids)
                res_all = pd.concat(res_chunks)
            else:
                res_all = pd.DataFrame()
                i = 1
                n = len(chunks_ids)
                for chunk_ids in chunks_ids:
                    if verbose:
                        print('Chunk '+str(i)+' / '+str(n), end='\r')
                        i += 1
                    res_chunk = self._compounded_return_for_id(chunk_ids, hol, dfu, window, min_periods,
                                                             logreturn, ascending, useall, tolerance)
                    res_all = pd.concat([res_all, res_chunk])

        # Return the variable for the user data
        return(res_all)

    def return_index(self, data, nations, sics):
        """ Return the average daily returns for the given data
        for the subset of firms defined by nations and sics.

        arguments:
            data        User data that includes at least a daily date column
            nations     List of nations to include
            sics        List of sic codes to include (how about sic2d vs sic4d?)

        """
        None

    def market_value(self, data):
        """ Fetch market values
        """
        # Use only the necessary columns from user data
        dfu = data[[self.col_id, self.d.col_date]]
        # Open the market value file
        dmv = self.open_data(self.DS2MktVal)
        # Merge to the user data
        dfin = dfu.merge(dmv, how='left',
                         left_on=[self.col_id, self.d.col_date],
                         right_on=[self.col_id, 'valdate'])
        dfin.index = dfu.index
        return dfin.consolmktval
