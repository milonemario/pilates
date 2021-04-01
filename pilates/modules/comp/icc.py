"""
Provides functions to compute several measures of the Implied Cost of Capital.
"""

from pilates import data_module
import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
# from functools import partial

################################################
# Global Function (needed for parallelization) #
################################################

def _get_root(x, fn):
    # Find all the roots
    rs, i, ier, msg = scipy.optimize.fsolve(fn, .05,
                                            args=(x),
                                            full_output=1)
    if ier == 1:
        if len(rs) == 1:
            return(rs[0])
        elif len(rs) == 0:
            return(np.nan)
        else:
            print("warning: multiple roots for the cost of capital")
            print(x)
            return(np.nan)
    else:
        return(np.nan)


def _get_root_gls(df):
    def fn(R, d):
        S = 0  # Sum
        for k in range(1, 12):
            ROE = 'ROE' + str(k)
            if k == 1:
                B = 'B'
            else:
                B = 'B' + str(k-1)
            S += ((d[ROE]-R)*d[B]) / ((1+R)**k)
        T = ((d.ROE12-R)*d.B11) / (R*(1+R)**11)  # Termination value
        return(d.B + S + T - d.M)
    def root(x):
        return(_get_root(x, fn))
    return(df.apply(root, axis=1))


def _get_root_ct(df):
    def fn(R, d):
        S = 0  # Sum
        for k in range(1, 6):
            ROE = 'ROE' + str(k)
            if k == 1:
                B = 'B'
            else:
                B = 'B' + str(k-1)
            S += ((d[ROE]-R)*d[B]) / ((1+R)**k)
        T = ((d.ROE5-R)*d.B4*(1+d.g)) / ((R-d.g)*(1+R)**5)  # Term. value
        return(d.B + S + T - d.M)
    def root(x):
        return(_get_root(x, fn))
    return(df.apply(root, axis=1))


def _get_root_gordon(df):
    def fn(R, d):
        return((d.E1 / R) - d.M)
    def root(x):
        return(_get_root(x, fn))
    return(df.apply(root, axis=1))


def _get_root_mpeg(df):
    def fn(R, d):
        return(((d.E2 + R * d.D1 - d.E1) / R**2) - d.M)
    def root(x):
        return(_get_root(x, fn))
    return(df.apply(root, axis=1))


class icc(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
        # Initialize values
        self.col_id = 'gvkey'
        self.col_date = 'datadate'

    def hdz_ct(self, data, freq='Q', ind_vars=None):
        """ Compute the cost of capital of Claus and Thomas (2001) following
        the methodology of Hou, van Dijk and Zhang (2012, JAE) paper.
        """
        df = self._earnings_forecasts_hdz(data, freq=freq,
                                          N=5, ind_vars=ind_vars)
        # Get the market prices
        df['M'] = self.d.comp.get_fields(['prccq'], df)
        # Get the growth rate g
        df['g'] = self.d.fred.get_series(df, ['DGS10'], col_date='datadate')
        #df['g'] = self.d.fred.get_10y_US_rates(df, col_date='datadate')
        df['g'] = df.g - 0.03
        if freq == 'Q':  # Transform the growth rate to quarterly if needed
            df['g'] = df.g / 4
        # Keep the rows with the relevant variables
        subset = ['M', 'B', 'ROE5']
        subset += ['B'+str(k) for k in range(1, 5)]
        subset += ['ROE'+str(k) for k in range(1, 5)]
        df = df.dropna(subset=subset)
        # Compute the cost of capital
        # Apply the function in parallel
        df_split = np.array_split(df[:1000], self.d.cores)
        pool = mp.Pool(self.d.cores)
        cc = pd.concat(pool.map(_get_root_ct, df_split))
        pool.close()
        pool.join()
        # Create the dataset to merge with the user data
        key = ['gvkey', 'datadate']
        dfc = df[key]
        dfc['icc'] = cc
        # Merge with the user data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(dfc, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.icc.astype('float32'))

    def hdz_gls(self, data, freq='Q', ind_vars=None):
        """ Compute the cost of capital of Gebhardt et al (2004) following
        the methodology of Hou, van Dijk and Zhang (2012, JAE) paper.
        """
        df = self._earnings_forecasts_hdz(data, freq=freq,
                                          N=12, ind_vars=ind_vars)
        # Get the market prices
        df['M'] = self.d.comp.get_fields(['prccq'], df)
        # Keep the rows with the relevant variables
        subset = ['M', 'B', 'ROE12']
        subset += ['B'+str(k) for k in range(1, 12)]
        subset += ['ROE'+str(k) for k in range(1, 12)]
        df = df.dropna(subset=subset)
        # Compute the cost of capital
        # Apply the function in parallel
        df_split = np.array_split(df, self.d.cores)
        pool = mp.Pool(self.d.cores)
        cc = pd.concat(pool.map(_get_root_gls, df_split))
        pool.close()
        pool.join()
        # Create the dataset to merge with the user data
        key = ['gvkey', 'datadate']
        dfc = df[key]
        dfc['icc'] = cc
        # Merge with the user data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(dfc, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.icc.astype('float32'))

    def hdz_gordon(self, data, freq='Q', ind_vars=None):
        """ Compute the cost of capital of Gordon and Gordon (1997) following
        the methodology of Hou, van Dijk and Zhang (2012, JAE) paper.
        """
        df = self._earnings_forecasts_hdz(data, freq=freq,
                                          N=1, ind_vars=ind_vars)
        # Get the market prices
        df['M'] = self.d.comp.get_fields(['prccq'], df)
        # Keep the rows with the relevant variables
        subset = ['E1', 'M']
        df = df.dropna(subset=subset)
        # Compute the cost of capital
        # Apply the function in parallel
        df_split = np.array_split(df, self.d.cores)
        pool = mp.Pool(self.d.cores)
        cc = pd.concat(pool.map(_get_root_gordon, df_split))
        pool.close()
        pool.join()
        # Create the dataset to merge with the user data
        key = ['gvkey', 'datadate']
        dfc = df[key]
        dfc['icc'] = cc
        # Merge with the user data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(dfc, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.icc.astype('float32'))

    def hdz_mpeg(self, data, freq='Q', ind_vars=None):
        """ Compute the cost of capital of Easton (2004) following
        the methodology of Hou, van Dijk and Zhang (2012, JAE) paper.
        """
        df = self._earnings_forecasts_hdz(data, freq=freq,
                                          N=2, ind_vars=ind_vars)
        # Get the market prices
        df['M'] = self.d.comp.get_fields(['prccq'], df)
        # Keep the rows with the relevant variables
        subset = ['E2', 'D1', 'E1', 'M']
        df = df.dropna(subset=subset)
        # Compute the cost of capital
        # Apply the function in parallel
        df_split = np.array_split(df, self.d.cores)
        pool = mp.Pool(self.d.cores)
        cc = pd.concat(pool.map(_get_root_mpeg, df_split))
        pool.close()
        pool.join()
        # Create the dataset to merge with the user data
        key = ['gvkey', 'datadate']
        dfc = df[key]
        dfc['icc'] = cc
        # Merge with the user data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(dfc, how='left', on=key)
        dfin.index = dfu.index
        return(dfin.icc.astype('float32'))

    def hdz_oj(self, data, freq='Q', ind_vars=None):
        """ Compute the cost of capital of Ohlson and Juettner-Nauroth (2005)
        following
        the methodology of Hou, van Dijk and Zhang (2012, JAE) paper.
        """
        df = self._earnings_forecasts_hdz(data, freq=freq,
                                          N=5, ind_vars=ind_vars)
        # Get the market prices
        df['M'] = self.d.comp.get_fields(['prccq'], df)
        # Get the growth rate g
        df['gamma'] = self.d.fred.get_series(df, ['DGS10'], col_date='datadate')
        #df['gamma'] = self.d.fred.get_10y_US_rates(df, col_date='datadate')
        df['gamma'] = df.gamma - 0.03
        if freq == 'Q':  # Transform the growth rate to quarterly if needed
            df['gamma'] = df.gamma / 4
        # Keep the rows with the relevant variables
        subset = ['M', 'D1', 'gamma']
        subset += ['E'+str(k) for k in range(1, 6)]
        df = df.dropna(subset=subset)
        # Compute the cost of capital
        df['A'] = 0.5 * ((df.gamma-1) + (df.D1/df.M))
        df['g'] = 0.5 * (((df.E3-df.E2)/df.E2) + ((df.E5-df.E4)/df.E4))
        df['icc'] = df.A + np.sqrt(df.A**2 + (df.E1/df.M)*(df.g-(df.gamma-1)))
        # Create the dataset to merge with the user data
        key = ['gvkey', 'datadate']
        # Merge with the user data
        dfu = self.open_data(data, key)
        dfin = dfu.merge(df[key+['icc']], how='left', on=key)
        dfin.index = dfu.index
        return(dfin.icc.astype('float32'))

    def _earnings_forecasts_hdz(self, data, freq='Q', N=5, ind_vars=None):
        """ Compute the cost of capital
        following Hou, van Dijk and Zhang (HDZ)
        (2012, Journal of Accounting and Economics) paper.
        Argments:
            data -- User provided data.
                    Required columns: ['gvkey', 'datadate']
            freq -- Frequency of the data
            N --    Defines how many future forecasts are needed
            ind_vars -- Independent variables to use in addition to the ones
                        in HDZ. These variables should be in the user data.
        """
        ##########################
        # Step 1: COMPUSTAT DATA #
        ##########################
        key = ['gvkey', 'datadate']
        fields_names = ['E', 'B', 'A', 'D', 'AC']
        # We may need more dates than is in the user data.
        # This code uses as much as possible from compustat
        # Add Compustat variables
        # TODO: Add _oacc() function for yearly data  in the comp module
        if freq == 'A':
            fields = ['ib', 'ceq', 'at', 'dvt', 'oacc']
        elif freq == 'Q':
            fields = ['ibq', 'ceqq', 'atq', 'dvtq', 'oaccq']
        dc = self.d.comp.get_fields(fields).drop_duplicates()
        dc.columns = key+fields_names
        # Add additional independent variables from the user data
        dfu = self.open_data(data, key + ind_vars).drop_duplicates()
        dc = dc.merge(dfu, how='left', on=key)

        ##########################################
        # Step 2: Cross-sectional earnings model #
        ##########################################
        # Run E[i,t+tau] = a0 + a1.A[i,t] + a2.D[i,t] + a3.DD[i,t]
        #                     + a4.E[i,t] + a5.NegE[i,t] + a6.AC[i,t]
        #                     + eps[i,t]
        # Where
        #   - E are earnings (ib)
        #   - A are total assets (at)
        #   - D are dividends (dvt)
        #   - DD = 1 if D > 0
        #   - NegE = 1 if E < 0
        #   - AC are accruals (oacc)
        #
        # tau in (1, 5)
        # For annual data, use 10 years of historical data.
        # For quarterly data, use r years of historical data
        dc['DD'] = 0
        dc.loc[dc.D > 0, 'DD'] = 1
        dc['NegE'] = 0
        dc.loc[dc.E < 0, 'NegE'] = 1
        ctrls = ['A', 'D', 'DD', 'E', 'NegE', 'AC'] + ind_vars
        # Save the dataset to make prediction at the end
        dc_final = dc.copy()
        # Windsorize at the 1st and 99th percentile for each time period
        # Only windsorize using hdz variables
        dates = dc.datadate.sort_values().drop_duplicates()
        winfields = ['E', 'B', 'A', 'AC']
        for d in dates:
            # Get the percentiles
            dd = dc[dc.datadate == d][winfields]
            low = dd.quantile(.01)
            high = dd.quantile(.99)
            dc.loc[dc.datadate == d, winfields] = dd.clip(low, high, axis=1)
        # Run the regressions for each time period and compute earnings
        # forecasts for up to 5 time periods into the future
        dc = dc.sort_values(key)
        for i in range(1, N+1):
            dc['E'+str(i)] = dc.groupby('gvkey').E.shift(-i)
        dc = dc.dropna()
        # Run OLS for every time period starting from 3 years
        # after the first date
        dates = dc.datadate.sort_values().drop_duplicates()
        dstart = dates.iloc[0] + relativedelta(years=3)
        dates = dates[dates >= dstart]
        coefs_df = pd.DataFrame(dates)
        for c in ctrls:
            coefs_df[c] = np.nan
        coefs = {}
        for i in range(1, N+1):
            coefs['E'+str(i)] = coefs_df.copy()
        # Run OLS and save the coefficients
        ols = linear_model.LinearRegression()
        for enddt in dates:
            startdt = enddt - relativedelta(years=3)
            df = dc[(dc.datadate >= startdt) & (dc.datadate <= enddt)]
            X = df[ctrls]
            for e in range(1, N+1):
                E = 'E'+str(e)
                y = df[E]
                c = ols.fit(X, y).coef_
                coefs[E].loc[coefs[E].datadate == enddt, ctrls] = c

        ##########################################
        # Step 3: Predictions of future earnings #
        ##########################################
        dcf = dc_final[dc_final.datadate >= dstart].dropna().copy()
        for e in range(1, N+1):
            dcf['E'+str(e)] = np.nan
        for date in dates:
            dd = dcf[dcf.datadate == date][ctrls]
            for e in range(1, N+1):
                E = 'E' + str(e)
                c = coefs[E][coefs[E].datadate == date][ctrls]
                est = np.array(dd.dot(c.transpose()))
                dcf.loc[dcf.datadate == date, E] = est

        ###############################################
        # Step 4: Compute necessary variables for icc #
        ###############################################
        # Future Dividend
        # D[t+k] = DivRatio[t] * E[t+k]
        # DivRatio[t] = Current Dividend Payout Ratio = D[t] / E[t] if D[t] > 0
        # DivRatio[t] = D[t] / (0.06 * A[t]) if E[t] < 0
        dcf['div_ratio'] = dcf.D / dcf.E
        dcf.loc[dcf.E < 0, 'div_ratio'] = dcf.D / (0.06 * dcf.A)
        for i in range(1, N+1):
            E = 'E' + str(i)
            D = 'D' + str(i)
            dcf[D] = dcf.div_ratio * dcf[E]
        dcf = dcf.drop('div_ratio', axis=1)
        # Future Book Equity
        # B[t+k] = B[t+k-1] + E[t+k] - D[t+k]
        for i in range(1, N+1):
            B = 'B' + str(i)
            if i == 1:
                Bl = 'B'
            else:
                Bl = 'B' + str(i-1)
            E = 'E' + str(i)
            D = 'D' + str(i)
            dcf[B] = dcf[Bl] + dcf[E] - dcf[D]
        # Retrun on Equity (ROE)
        # ROE[t+k] = E[t+k] / B[t+k]
        for i in range(1, N+1):
            ROE = 'ROE' + str(i)
            E = 'E' + str(i)
            B = 'B' + str(i)
            dcf[ROE] = dcf[E] / dcf[B]

        ###########################################
        # Step 5: Return the earnings predictions #
        ###########################################
        # We now open the user data and append the ICC measure
        df = self.open_data(data, key)
        index = df.index
        cols = ['E', 'B']
        for v in ['E', 'D', 'B', 'ROE']:
            cols += [v+str(i) for i in range(1, N+1)]
        dfin = df.merge(dcf[key+cols], how='left', on=key)
        dfin.index = index
        return(dfin)
