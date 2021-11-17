"""
Module to provide processing functions for Compustat data.

"""

from pilates import wrds_module
import pandas as pd
import numpy as np
import country_converter as coco


class worldscope(wrds_module):
    """ Class providing main processing methods for the Worldscope data.
    """

    def __init__(self, d):
        wrds_module.__init__(self, d)
        # Initialize values
        self.col_id = 'item6105'
        #self.col_date = 'item5350'
        #self.key = [self.col_id, self.col_date]
        self.freq = None  # Data frequency
        # Filters
        self.use_restated = True

    # def set_date_column(self, col_date):
    #     self.col_date = col_date
    #     self.key = [self.col_id] + self.col_date

    def set_frequency(self, frequency):
        """ Define the frequency to use.
        """
        if frequency in ['Quarterly', 'quarterly', 'Q', 'q']:
            self.freq = 'Q'
            self.fund = self.wrds_ws_fundq
            self.col_date = ['year_', 'seq']
        elif frequency in ['Annual', 'annual', 'A', 'a',
        'Yearly', 'yearly', 'Y', 'y']:
            self.freq = 'A'
            self.fund = self.wrds_ws_funda
            self.col_date = ['year_']
        else:
            raise Exception('Compustat data frequency should by either',
                            'Annual or Quarterly')
        self.key = [self.col_id] + self.col_date

    def infocode_from_wsid(self, data):
        """ Add Datastream infocode to worldscope usind wsid (item6105)
        """
        # 1-1 mapping between WS code and DS infocode
        code = self._get_company_fields(['code'], data)
        return(code.astype('Int32'))

    def _get_fund_fields(self, fields, data, lag=0):
        """ Returns fields from Worldscope 'fund' file.
        'funda' or 'fundq' depending on the frequency chosen.
        """
        key = self.key
        cols_req = []
        if self.freq=='A':
            cols_req += ['freq']
        data_key = self.open_data(data, key)
        index = data_key.index
        df = self.open_data(self.fund, key+fields+cols_req).drop_duplicates()
        # Filter the fund data
        if self.freq=='A' and self.use_restated:
            # Use restated data when available
            # Use non-null values from restated info first
            df = df.set_index(key)
            dfA = df[df.freq=='A']
            dfB = df[df.freq=='B']
            df = dfB.combine_first(dfA).reset_index()
        df = df.sort_values(key)
        # Get the lags if asked for
        if lag != 0:
            dfl = df.groupby(self.col_id)[fields].shift(lag)
            df[fields] = dfl
        # Merge and return the fields
        dfin = data_key.merge(df, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def _get_stock_fields(self, fields, data, lag=0):
        """ Returns fields from Worldscope 'stock' file.
        """
        key = self.key
        data_key = self.open_data(data, key)
        index = data_key.index
        df = self.open_data(self.wrds_ws_stock, key+fields).drop_duplicates()
        # Get the lags if asked for
        df = df.sort_values(key)
        dfl = df.groupby(self.col_id)[fields].shift(lag)
        df[fields] = dfl
        # Merge and return the fields
        dfin = data_key.merge(df, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def _get_company_fields(self, fields, data):
        """ Returns fields from COMPUSTAT 'company' file.
        """
        key = [self.col_id]
        #cols_req = ['freq']
        data_key = self.open_data(data, key)
        index = data_key.index
        df = self.open_data(self.wrds_ws_company,
                            key+fields).drop_duplicates()
        # Clean the company file
        ## Keep non-null keys
        df = df[df[self.col_id].notnull()]
        ## Remove duplicates
        dup = df[key].duplicated(keep=False)
        nd0 = df[~dup]
        dd = df[dup][key+fields]
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
        # Merge and return the fields
        dfin = data_key.merge(nd, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def _get_segments_fields(self, fields, data):
        """ Returns fields from COMPUSTAT 'company' file.
        """
        key = self.key
        data_key = self.open_data(data, key)
        index = data_key.index
        df = self.open_data(self.wrds_ws_segments,
                            key+fields+['freq']).drop_duplicates()
        # Clean the company file
        ## Keep non-null keys
        for k in key:
            df = df[df[k].notnull()]
        ## Remove duplicates
        dup = df[key].duplicated(keep=False)
        nd0 = df[~dup]
        # 1. Choose between restated or non-restated
        dd = df[dup][key+['freq']+fields]
        if self.use_restated:
            freq = 'B'
        else:
            freq = 'A'
        ddA = dd[dd.freq=='A'][key+fields].set_index(key)
        ddB = dd[dd.freq=='B'][key+fields].set_index(key)
        dd = ddB.combine_first(ddA).reset_index()
        dup = dd[key].duplicated(keep=False)
        nd1 = dd[~dup][key+fields]
        dd = dd[dup]
        # 2. When duplicates, keep the ones with the most information
        dd['n'] = dd.count(axis=1)
        maxn = dd.groupby(key).n.max().reset_index()
        maxn = maxn.rename({'n': 'maxn'}, axis=1)
        dd = dd.merge(maxn, how='left', on=key)
        dd = dd[dd.n == dd.maxn]
        dup = dd[key].duplicated(keep=False)
        nd2 = dd[~dup][key+fields]
        dd = dd[dup]
        # 3. Choose randomly
        dd['rn'] = dd.groupby(key).cumcount()
        dd = dd[dd.rn == 0]
        nd3 = dd[key+fields]
        # Concatenate the non duplicates
        nd = pd.concat([nd0, nd1, nd2, nd3]).sort_values(key)
        # Merge and return the fields
        dfin = data_key.merge(nd, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def _get_info_fields(self, fields, data):
        """ Returns fields from COMPUSTAT 'company' file.
        """
        key = [self.col_id]
        key_info = ['wsid']

        #cols_req = ['freq']
        data_key = self.open_data(data, key)
        index = data_key.index
        df = self.open_data(self.wsinfo,
                            key_info+fields).drop_duplicates()
        df.rename(columns={key_info[0]: key[0]}, inplace=True)
        # Clean the company file
        ## Keep non-null keys
        df = df[df[self.col_id].notnull()]
        # Merge and return the fields
        dfin = data_key.merge(df, how='left', on=key)
        dfin.index = index
        return(dfin[fields])

    def get_fields(self, fields, data=None, lag=0):
        """ Returns fields from Worldscope data.
        """
        key = self.key
        # fields_toc = [f for f in fields if (f not in fund_raw and
        #                                     f not in names_raw)]
        # Keep the full worldscope key (to correctly compute lags)
        df = self.open_data(self.fund, key).drop_duplicates()
        # Keep non-null keys
        for k in key:
            df = df[df[k].notnull()]
        # Determine the raw and additional fields
        cols_fund = self.get_fields_names(self.fund)
        cols_stock = self.get_fields_names(self.wrds_ws_stock)
        cols_company = self.get_fields_names(self.wrds_ws_company)
        cols_segments = self.get_fields_names(self.wrds_ws_segments)
        cols_info = self.get_fields_names(self.wsinfo)
        fields = [f for f in fields if f not in key]
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
        stock_raw = [f for f in fields if (f in cols_stock and
                                          f not in fields_add)]
        company_raw = [f for f in fields if (f in cols_company and
                                           f not in fields_add)]
        segments_raw = [f for f in fields if (f in cols_segments and
                                           f not in fields_add)]
        info_raw = [f for f in fields if (f in cols_info and
                                           f not in fields_add)]
        # Get the raw fields for the different files
        if len(fund_raw) > 0:
            df[fund_raw] = self._get_fund_fields(fund_raw, df)
        if len(stock_raw) > 0:
            df[stock_raw] = self._get_stock_fields(stock_raw, df)
        if len(company_raw) > 0:
            df[company_raw] = self._get_company_fields(company_raw, df)
        if len(segments_raw) > 0:
            df[segments_raw] = self._get_segments_fields(segments_raw, df)
        if len(info_raw) > 0:
            df[info_raw] = self._get_info_fields(info_raw, df)
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

    def get_segments_data(self):
        ## Create the fields dictionary
        # ID info
        fields = {
            'item6026': 'nation',
            'isonation': 'country_out'
                  }
        # Geographic segments
        for gsi in range(0, 10):
            fields['item196'+str(gsi)+'0'] = 'segment_description'+str(gsi+1)
            fields['item196'+str(gsi)+'1'] = 'segment_sales'+str(gsi+1)
            fields['item196'+str(gsi)+'2'] = 'segment_oic'+str(gsi+1)
            fields['item196'+str(gsi)+'3'] = 'segment_assets'+str(gsi+1)
            fields['item196'+str(gsi)+'4'] = 'segment_capex'+str(gsi+1)
            fields['item196'+str(gsi)+'5'] = 'segment_depreciation'+str(gsi+1)

        fields_keys = list(fields.keys())
        fields_names = list(fields.values())
        data = self.get_fields(fields_keys)
        data.columns = self.key + fields_names
        data = self.clean_segments(data)
        data = self._correct_columns_types(data)
        return(data)

    def clean_segments(self, data):
        # Reshape segments data
        stubnames = ['segment_description',
                     'segment_sales',
                     'segment_oic',
                     'segment_assets',
                     'segment_capex',
                     'segment_depreciation']
        dfl = pd.wide_to_long(data, stubnames, i=self.key, j='segment')
        # Clean data
        dflc = dfl[(dfl.segment_description.notnull()) & (dfl.nation.notnull())]

        ########################################
        # Clean segment countries (country IN) #
        ########################################

        # Clean Names
        dflc.segment_description = dflc.segment_description.str.upper()
        remove_list = ['(COUNTRY)', '(REGION)', '(PROVINCE)', '??']
        for remove in remove_list:
            dflc.segment_description = dflc.segment_description.str.replace(remove,'', regex=False)
            dflc.segment_description = dflc.segment_description.str.strip()
        dflc = dflc[dflc.segment_description != '']

        # NOTE: Domestic can be mispelled: [DOMESTIC, DOEMSTIC]

        # Extract countries info
        des = pd.DataFrame(dflc.reset_index().segment_description.drop_duplicates().reset_index(drop=True))

        ## Standardize names for segments (country IN)
        des['standardized'] = coco.convert(names=des.segment_description, to='name_short')

        ## Hand cleaning of countries not identified
        if True:
            #des2 = des[des.standardized=='not found']
            #des2.to_csv('descripions_todo.csv', index=False)
            countries = {}
            countries['Afghanistan'] = ['AFGANISTAN']
            countries['Algeria'] = ['ARGELIA']
            countries['Angola'] = ['REPUBLIC OF ANGOLIA']
            countries['Argentina'] = ['NEUQUINA BASIN', 'CUYANA BASIN']
            countries['Australia'] = ['VICTORIA', 'QUEENSLAND & OTHER', 'NEW SOUTH WALES', 'SYDNEY',
                         'MELBOURNE', 'BRISBANE', 'QUEENSLAND', 'AUTRALIA', 'AUATRALIA',
                         'TASMAN', 'QUEENSLAND.', 'QUEENSLAND GALILEE',
                         'QUEENSLAND BOWEN', 'NEW SOUTH WALES GUNNEDAH', 'QUEENSLAND (STATE)',
                         'NEW SOUTH WALES (STATE)', 'VICTORIA (STATE)', 'AUSTRALAIA',
                         'AUSTALIA', 'TASMANIA', 'CANBERRA', 'PERTH']
            countries['Azerbaijan'] = ['AZERBAYCAN']
            countries['Bahamas'] = ['BAHAMA']
            countries['Bahrain'] = ['BAHARAIN', 'BAHARIN']
            countries['Bangladesh'] = ['BANGALDESH', 'BASNGLADESH']
            countries['Barbados'] = ['BARBADO', 'BARBODOS', 'BARBODAS']
            countries['Belarus'] = ['BELAUS', 'BELORUSSIA']
            countries['Belgium'] = ['BELGIUM & LUXEMBOURG', 'WALLONIA', 'BRUSSELS', 'FLANDERS',
                       'WALLON REGION', 'FLEMISH REGION', 'BELUX',
                       'BRUSSELS DECENTRALISED', 'BRUSSELS CBD', 'BRUSSELS PERIPHERY',
                       'ANTWERP', 'BELGUIM', 'WALLONNE', 'BRUSSELS CENTRE (CBD)',
                       'BRUSSELS AIRPORT', 'BELGIUM/LUXEMBOURG', 'BELGIUM AND LUXEMBOURG',
                       'BELGIUM AND LUXEMBOURG (ECONOMICS)', 'BELGIUM & LUXEMBURG',
                       'BELGIUM / LUXEMBOUR', 'BELGIUM / LUXEMBOURG', 'BELGIUM/LUXEMBURG',
                       'BELGIUM-LUXEMBOURG', 'BELGIUM AND LUXEMBURG', 'BELGIA',
                       'BELGIUM /LUXEMBOURG']
            countries['Bosnia an Herzegovina'] = ['BANJA LUKA']
            countries['Botswana'] = ['BOSTWANA']
            countries['Brazil'] = ['BRASIL', 'BARZIL']
            countries['Bulgaria'] = ['BOULGARIA', 'SOFIA', 'BOLGARIA', 'BULGIRIA']
            countries['Cambodia'] = ['CAMBODGE', 'COMBODIA', 'CANBODIA']
            countries['Cabo Verde'] = ['COBO  VERDE']
            countries['Cameroon'] = ['CAMEROUN']
            countries['Canada'] = ['CANDA', 'ALBERTA', 'ONTARIO', 'QUEBEC', 'BRITISH COLUMBIA',
                      'CANADIAN OILFIELD SERVICES', 'CANADIAN OPERATIONS',
                      'CANANDA', 'NUNAVUT', 'NEWFOUNDLAND', 'NORTHWEST TERRITORIES',
                      'CANADIAN', 'VANCOUVER', 'ONTARIO(ON)', 'ALBERTA(AB)',
                      'BRITISH COLUMBIA(BC)']
            countries['Chile'] = ['CHILIE']
            countries['China'] = ['PRC', 'THE PRC', 'CHINESE MAINLAND', 'OUTSIDE SHANGHAI AREA',
                     'PRC REGIONS', 'WITHIN PRC', 'P.R.C.', 'PRC MAINLAND',
                     'OTHER REGIONS IN THE PRC', 'OTHER REGION IN THE PRC',
                     'PRC EXCLUDING HONG', 'OTHER REGIONS OF THE PRC',
                     'PRC (THE MAINLAND)', 'ELSEWHERE IN PRC', 'ELSEWHERE IN THE PRC',
                     "PEOPLE'S REPUBLIC OF C", 'MAINLAND PRC',
                     'THE  PRC', 'ELSWHERE IN THE PRC', 'GAUANGZHOU', 'OTHER REGION IN PRC',
                     'OTH REGIONS OF PRC', 'OTHER PARTS OF PRC', 'OTHER PART OF PRC',
                     'CHINESE', 'CHAINA', 'SHANGHAI', "PEOPLE'S REPUBLIC OF CH",
                     'GUANGDONG', 'REST OF PRC', 'FOSHAN', 'CHONGQING', 'NANJING AND JURONG',
                     'GUANGZHOU', 'XI AN', 'GUANGDONG PROVINCE IN THE PRC', 'HENAN',
                     'GANSU', 'BEIJING (MUNICIPALITY)', 'SICHUAN', 'DALIAN AREA',
                     'SHENGWAI', 'SHENGNEI', 'LIAONING', 'CHIMA', 'IN THE PRC',
                     'PRC EASTERN REGION', 'PRC SOUTHERN REGION', 'PRC NORTHERN REGION',
                     'PRC NORTHWEST REGION', 'PRC SOUTHWEST REGION', 'PRC CENTRAL REGION',
                     'PRC NORTHEAST REGION', 'REVENUE FROM PRC', 'BEIJING',
                     'NORTHERN JIANGSU PROVINCE', 'SOUTHERN JIANGSU PROVINCE',
                     'WESTERN REGION OF THE PRC', 'SOUTHERN JIANGSU PROVIANCE',
                     'HENAN LUONING', 'HUNAN', 'JIANGSU', 'ZHEJIANG', 'SINKIANG',
                     'GUANGXI', 'SHANDONG PROVINCE', 'SHANXI PROVINCE', 'XINJIANG PROVINCE',
                     'HUBEI', 'SHANDONG', 'YUNNAN', 'FUJIAN(COMPANY)', 'JIANGXI',
                     'HENAN & OTHER', 'YANGTZE RIVER DELTA', 'BOHAI RIM', 'PRC (COUNTRY OF DOMICILE)',
                     'PRC MARKET', 'FUJIAN PROVINCE', 'GUANGDONG PROVINCE',
                     'THE PRC, EXCLUDING', 'NORTHERN, PRC', 'MIDDLE SOUTH, PRC',
                     'THE PRC MAINLAND', 'JINZHOU', 'GUANGXI (AUTONOMOUS REGION)',
                     'JILIN REGION', 'SHANDONG PROVINCE OUTSIDE', 'THE PRC, EXCLUDING SARS',
                     'DOMESTIC PRC REVENUES', 'FUJIAN'
                     # Not so precise
                     #,'PRC INCLUDING HONG KO', 'PRC AND OTHERS', 'PRC, INCLUDING HKG',
                     #'PRC INLC HK', 'PRC INCLUDING HONGKON', 'ASIA - CHIN',
                     #'REST OF PRC AND OTHERS', 'PRC (INCLUDING HK)', 'PRC & HK',
                     #'THE PRC OTHER THAN', 'REST OF PRC & OTHER', 'PRC & OTHER'
                     ]
            countries['Colombia'] = ['COLUMBIA', 'COLUMBIA & OTHER']
            countries['Congo Republic'] = ['REPUBLIC OF THE CONGA', 'THE REPUBLIC OF CONGO'
                     # Not so precise
                     #,'CONGO & OTHER', 'THE REPUBLIC OF CONGO & OTHER'
                     ]
            countries['DR Congo'] = ['DRC', 'RDC']
            countries['Costa Rica'] = ['COST RICA']
            countries["Cote d'Ivoire"] = ["COTE D'LVORIE", 'VORY COAST']
            countries['Croatia'] = ['CROACIA', 'CROTATIA', 'ZAGREB', 'CROTIA']
            countries['Czech Republic'] = ['PRAGUE']
            countries['Denmark'] = ['COPENHAGEN', 'GREEN LAND', 'DANISH GENERAL INSURANCE',
                       'DANIA']
            countries['Dominican Republic'] = ['DOM REPUBLIC', 'DOM. REPUBLIC', 'DOMINION REPUBLIC']
            countries['Ecuador'] = ['OTHER/EQUATOR RE', 'EQUATOR RE', 'ECUADR', 'ECUDOR', 'EQUADOR']
            countries['El Salvador'] = ['SALVADOR'
                          # Not so precise
                          #, 'SALVADOR & OTHER'
                          ]
            countries['Ethiopia'] = ['ETHOPIA']
            countries['Finland'] = ['FINNLAND', 'RAUMA', 'HELSINGFORS', 'TIKKURILA COMMON']
            countries['France'] = ['FRENCH TERRITORIES OVERSEAS', 'PARIS', 'PARIS REGION OUTSIDE PARIS',
                      'PROVINCE', 'OUTSIDE PARIS', 'PARIS & SUBURBS', 'BURGUNDY',
                      'FRANCE & FRENCH OVERSEAS DEPARTMENTS', 'FRENCH OVERSEAS TERRITORIES',
                      'RH NE ALPES (METROPOLITAN REGION)', 'NORD-PAS-DE-CALAIS (METROPOLITAN REGION)',
                      "PROVENCE-ALPES-C TE-D'AZUR (METROPOLITAN REGION)",
                      'BOURGOGNE (METROPOLITAN REGION)', 'RHONE ALPE',
                      'NORD PAS DE CALAIS', 'PACA', 'BOURGOGNE', 'RHONE-ALPES',
                      'FRENCH OVERSEAS DOMINIONS & TERRITORIES', 'FRENCH TERRITORIES',
                      'PARIS REGION', 'FRANCE - OVERSEAS DEPARTMENTS',
                      'FRENCH OVERSEAS TERRITORIES DOM', 'DOM TOM', 'MAINLAND FRNACE',
                      'OVERSEAS FRENCH TERRITORIES', 'MAINLAND FRANCE  OVERSEAS DEPARTMENTS AND TERRITORIES',
                      'MAINLAND FRANCE + OVERSEAS DEPARTMENTS AND TERRITORIES', 'FRENCH',
                      'FRANCH', 'PARIS AREA OUTSIDE PARIS', 'DOM-TOM', 'FRENCH GUYANE'
                      # Not so precise
                      ,'PIXMANIA'
                      ]
            countries['Gambia'] = ['GAMIBIA']
            countries['Germany'] = ['GERMAN', 'EASTERN GERMANY', 'WESTERN GERMANY', 'DEUTSCHLAND',
                       'GERMANY WEST', 'WEST-GERMANY', 'GERMAN MARKET',
                       'EUROPE (PRINCIPALLY WEST GERMANY)', 'EUROPE (PRINCIPALLY GERMANY)',
                       'EUROPE - WEST GERMANY', 'EUROPE - GERMANY', 'GERMAMY', 'GEMANY',
                       'NIEMCY', 'EUROPE -GERMANY', 'EUROPE (GERMANY)']
            countries['Greece'] = ['GREEK COASTAL', 'GREEK', 'GEECE', 'GERMAN OPERATIONS']
            countries['Guatemala'] = ['GUATEMAL', 'GUATMALA', 'GUETAMALA', 'GUATEMELA']
            countries['Honduras'] = ['HONDURUS']
            countries['Hong Kong'] = ['HONG HONG', 'HK SAR', 'KOWLOON'
                        # Not so precise
                        #,'HK & OTHERS', 'THE PRC (HK)', 'HK & AOMEN',
                        #'FAR EAST  HK/PRC', 'HK AND OTHERS'
                        ]
            countries['Hungary'] = ['HUNGARIAN', 'BUDAPEST', 'WEGRY']
            countries['India'] = ['INIDA', 'INDAI', 'CHHATTISGARH', 'TELANGANA & ANDHRA PRADESH']
            countries['Indonesia'] = ['JAKARTA & SAMARINDA', 'GRESIK', 'JAVA & BALI', 'JAWA',
                         'KALIMANTAN,SULAWESI & MALUKU', 'SUMATERA', 'JAVA',
                         'GAJAH MADA','TANGERANG', 'SERANG', 'CAKUNG', 'PONDOK CABE',
                         'BEKASI', 'SURABAYA', 'JAKARTA', 'BALI', 'SULAWESI',
                         'KALIMANTAN', 'JAWA BALI', 'JABOTABEK', 'MEDAN', 'MAKASSAR',
                         'SAMARINDA', 'PONTIANAK', 'BANDUNG', 'JAWA, BALI DAN NUSA TENGGARA',
                         'SULAWESI DAN PAPUA', 'BENGKULU', 'NORTH SUMATERA', 'RIAU',
                         'SOUTH SUMATERA', 'BANGKA', 'DKI JAKARTA', 'OUTSIDE DKI JAKARTA',
                         'EAST JAVA AND BALI', 'CENTRAL JAVA', 'JAKARTA AND BOGOR',
                         'BANTEN', 'MANADO', 'INDONSIA', 'JAVA ISLAND', 'JAWA (AREA)',
                         'KALIMANTAN (AREA)', 'SUMATERA (AREA)', 'WEST JAVA',
                         'KAMBUNA', 'INDONESHIA', 'SEMARANG', 'BALI AND NUSA TENGGARA',
                         'KUPANG', 'EAST JAVA', 'JABODETABEK', 'PASURUAN', 'PALEMBANG',
                         'PURWAKARTA', 'CIKANDE', 'JAWA-EXCLUDING JABODETABEK',
                         'JAVA ISLAND (EXC. JAKARTA)', 'SULAWESI AND MALUKU',
                         'BALI AND LOMBOK ISLAND', 'JAYAPURA', 'EAST JAVA - BALI',
                         'WEST KALIMANTAN', 'WEST SUMATERA', 'LAMPUNG', 'SULAWESI (AREA)',
                         'DENPASAR' 'JAWA (EXCLUDING JAKARTA)', 'JAVA (EXC. WEST JAVA)',
                         'NUSA TENGGARA', 'MALUKU & OTHER', 'MALUKU']
            countries['Ireland'] = ['IRLAND', 'EIRE', 'IRISH BUSINESS', 'REPUBLIC OF RELAND',
                       'REPULBLIC OF IRLAND'
                       # Not so precise
                       #,'IRELAND & NORTHERN IRELAND', 'IRELAND AND NORTHERN IRELAND'
                       ]
            countries['Israel'] = ['ISREAL']
            countries['Italy'] = ['ITLAY', 'WLOCHY', 'LOMBARDIA (AREA)', 'LIGURIA (AREA)',
                     'EMILIA-ROMAGNA (AREA)', 'TOSCANA (AREA)', 'VENETO (AREA)',
                     'LAZIO (AREA)', 'FRIULI-VENEZIA GIULIA (AREA)', 'PIEMONTE (AREA)',
                     'TRENTINO-ALTO ADIGE (AREA)', 'MARCHE (AREA) & OTHER']
            countries['Japan'] = ['JAPNA', 'JAPAM', 'JAPAC']
            countries['Kazakhstan'] = ['KAZACHSTAN']
            countries['Kenya'] = ['NAIROBI', 'NAIROBI REGION']
            countries['Kyrgyz Republic'] = ['KYRGYSZTAN', 'KYRGHYZTAN']
            countries['Lebanon'] = ['LIBAN', 'LIBANON']
            countries['Libya'] = ['LIBIA', 'LYBIA']
            countries['Lithuania'] = ['LITHUNIA', 'LITHUENIA', 'LITWA','LITHUANA']
            countries['Macedonia'] = ['MACENDONIA']
            countries['Madagascar'] = ['MADAGASKAR', 'MADAGAS']
            countries['Malawi'] = ['MALWAI']
            countries['Malaysia'] = ['NALAYSIA', 'MALAYASIA', 'MALYASIA', 'MALYSIA', 'KOTA KINABALU',
                        'KUCHING', 'KOTA BHARU', 'PASIR GUDANG', 'ALOR SETAR',
                        'SABAH & SARAWAK']
            countries['Mauritania'] = ['MAUTITANIA', 'MAURITANIE']
            countries['Mauritius'] = ['MAUTITIUS']
            countries['Mexico'] = ['MEXCIO', 'GUADALAJARA', 'MAXICO']
            countries['Mongolia'] = ['MANGOLIA', 'MONGOL']
            countries['Montenegro'] = ['BUDVA']
            countries['Morocco'] = ['MORROCCO', 'MARRUECOS']
            countries['Mozambique'] = ['MOZAMBICAN' 'TIE MAMBOFIVE', 'MOZANBIQUE']
            countries['Myanmar'] = ['GWA', 'MYAMAR']
            countries['Netherlands'] = ['NETHERLAND', 'HOLLAND', 'HOLLAND & OTHER', 'AMSTERDAM',
                           'HOLAND', 'NETHRLANDS', 'NETHERTLANDS', 'NETHELANDS',
                           'THE NETHERLAND', 'NEATHERLANDS', 'HOLLAND/NETHERLAND',
                           'WESTERN EUROPE-HOLLAND', 'NETHERANDS', 'DUTCH'
                           # Not so precise
                           #, 'NETHERLAND & OTHER'
                           ]
            countries['New Zealand'] = ['NEW ZELAND']
            countries['Nicaragua'] = ['MANAGUA'
                         # Not so precise
                         #, 'MANAGUA & OTHER'
                         ]
            countries['Norway'] = ['OSLO', 'NORWEGIAN GENERAL INSURANCE', 'NORWEGIA','NORAWAY',
                      'FINNMARK', 'ROGALAND']
            countries['Pakistan'] = ['PAKISTHAN']
            countries['Papua New Guinea'] = ['PAPUA']
            countries['Philippines'] = ['PHILLIPINES', 'PILIPIN', 'METRO MANILA',
                           'SOUTHERN LUZON', 'NORTHERN LUZON', 'MINDANAO',
                           'VISAYAS', 'PHILIPINES', 'PHILIPPINE', 'PHLIPPINES',
                           'PHILLIPPINES', 'CEBU', 'ILOILO', 'DAVAO']
            countries['Poland'] = ['POLLAND'
                      # Not so precise
                      #,'POLONIA & OTHER'
                     ]
            countries['Portugal'] = ['MADEIRA']
            countries['Romania'] = ['ROMANIS', 'ROMENIA', 'BUCHAREST']
            countries['Russia'] = ['SOVIET UNION', 'USSR', 'RUSIA', 'WESTERN SIBERIA',
                      'REPUBLIC OF TATARSTAN', 'ST. PETERSBURG',
                      'KOMI REPUBLIC', 'FORMER SOVIET UNION', 'U.S.S.R.',
                      'UNION OF SOVIET SOCIALIST REPUBLICS', 'RUSSSIA',
                      'URAL', 'CHELYABINSK REGION', 'URAL REGION',
                      'MOSCOW AND  MOSCOW RE', 'MOSCOW REGION', 'MOSCOW',
                      'KURSK', 'LIPETSK', 'VORONEZH', 'TAMBOV',
                      'LIPETSK REGION','KURSK REGION', 'VORONEZH REGION',
                      'TAMBOV REGION', 'SIBERIA', 'SOVIET REPUBLIC',
                      'ST.PETERSBURG', 'STAVROPOL', 'OMOLON', 'DUKAT',
                      'AMURSK ALBAZINO', 'VORO', 'OKHOTSK', 'MAYSKOYE',
                      'KYZYL', 'KHABAROVSK', 'MAGADAN',
                      'KALININGRADSKAYA OBLAST (ADMINISTRATIVE AREA)',
                      'MAGADAN BUSINESS UNIT', 'KRASNOYARSK BUSINESS UNIT',
                      'IRLUTSK ALLUVIAL BUSINESS UNIT', 'YAKUTSK KURANAKH BUSINESS UNIT',
                      'IRKUTSK ORE BUSINESS UNIT'
                      # Not so precise
                      #, 'UNION OF SOVIET SOCIALIST REPUBLICS  & OTHER'
                      ]
            countries['Saudi Arabia'] = ['ARAB SAUDI', 'SAUDI', 'SOUDI ARABIA', 'KINGDOM OF SAUDI ARAB',
                          'KINGDOME  OF SAUDI', 'SAUDI  ARABIA', 'SAUDI ARBIA',
                          'SOUDI  ARABIA', 'SAUDI AERABIA', 'MAKKAH', 'TABUK'
                          'KINGDOM OF SAUDI ARA', 'SAUDI ARAB']
            countries['Serbia'] = [
                           'BELGRADE', 'NOVI SAD', 'NIS', 'KRAGUJEVAC',
                           'BEOGRAD', 'KOMBANK INVEST AD BELGRADE'
                           # Not so precise
                           #,'SERBIA MONTENEGRO', 'SERBIA AND MONTENEGRO',
                           #'SERBIA MONTENEGRO & OTHER'
                           ]
            countries['Singapore'] = ['SINAPORE', 'SINGAOPORE', 'SINGAPORT', 'SINGAPOE',
                         'SINGPORE', 'SINGAPRE', 'SINAGPORE']
            countries['Slovakia'] = ['SOLVAKIA']
            countries['Slovenia'] = ['SLOVANIA', 'SOLVANIA']
            countries['South Africa'] = ['KWAZULU-NATAL', 'WESTERN CAPE', 'GAUTENG',
                            'EASTERN CAPE', 'COMZTEK AFRICA', 'REST OF SOUTH AFRIC',
                            'JOHANNESBURG', 'SANDTON', 'PRETORIA', 'DURBAN',
                            'ALBERTON', 'SOUT AFRICA', 'KWAZULU NATAL']
            countries['South Korea'] = ['SOUTH KORES' 'SOUTHKOREA', 'JEJU ISLAND']
            countries['Spain'] = ['SPANISH MARKET', 'SPAIM', 'CANARY ISLANDS']
            countries['Sri Lanka'] = ['SRI LNKA', 'UPCOT', 'MASKELIYA', 'BANDARAWELA', 'TALAWAKELLE',
                        'SRI LANAKA']
            countries['Sudan'] = ['SUNDAN']
            countries['Sweden'] = ['SOUTHERN STOCKHOLM', 'LIDINGO', 'WESTERN STOCKHOLM',
                      'HUDDINGE', 'STOCKHOLM', 'GOTHENBURG', 'ORESUND',
                      'MALARDALEN', 'OSTRA GOTALAND', 'STOCKHOLM NORTH',
                      'SWEDAN', 'LUND', 'MALM', 'SZWECJA', 'STOCKHOLM AREA']
            countries['Switzerland'] = ['SWIZERLAND', 'SWIDZERLAND', 'SWITERLAND', 'SWEEDEN',
                           'SWIZARLAND', 'SWIZTERLAND']
            countries['Syria'] = ['SIRYA', 'SIRIA'
                     # Not so precise
                     #, 'SIRIA & OTHER'
                     ]
            countries['Tajikistan'] = ['TAZAKISTAN']
            countries['Tanzania'] = ['ZANZIBAR', 'BULYANHULU', 'NORTH MARA', 'TULAWAKA', 'BUZWAGI']
            countries['Thailand'] =['BANGKOK', 'TAILAND', 'THAI', 'THAILLAND']
            countries['Timor-Leste'] = ['TIMOR']
            countries['Trinidad and Tobago'] = ['TRINDAD', 'TRINADAD']
            countries['Tunisia'] = ['TUNISIE', 'TUNIS']
            countries['Turkey'] = ['TURKISH MRARKET', 'TURQUIA', 'TURKISH REPUBLIC', 'TURKY']
            countries['Uganda'] = ['UGAMDA']
            countries['United Arab Emirates'] = ['DUBAI', 'DUBAI (EMIRATE)', 'DUBAI, UAE', 'U A E',
                   'U.A.E (DUBAI)', 'UAE MARKET', 'ABU DHABI (EMIRATE)',
                   'DUBAI UAE', 'ABU DHABI', 'KASHISH WORLDWIDE FZE, U.A.E.',
                   'DUBAI (UAE)', 'U.E.A'
                   # Not so precise
                   #, 'DUBAI & OTHER', 'UAE & ASIA', 'UAE AND ASIA',
                   #'UAE AND OTHERS', 'ABU DHABI (EMIRATE) & OTHER'
                   ]
            countries['United Kingdom'] = ['UK', 'NORTHERN IRELAND',
                  'ENGLAND', 'GREAT BRITIAN',
                  'U.K', 'UK & IRLAND', 'BRITISH ISLES',
                  'EGIDE UK', 'UNITED  KINGDOM', 'UK & ISLANDS', 'UK & ANGLO-NORMAND ISLAND',
                  'UK & ANGLO-NORMAND IS', 'STERLING ZONE',
                  'GBP', 'BRITISH POUND ZONE', 'U. K', 'THE UK', 'SCOTLAND',
                  'UK & IOM', 'UK & CHANNEL ISLANDS', 'LONDON', 'ENGLAND & WALES',
                  'FRANCHISING (UK)', 'INTERNATIONAL (UK)', 'UNITED  KINDOM', 'UNITED KINGDON',
                  'CONSTRUCTION - UK', 'UK RAIL', 'UK BUS (REGIONAL OPERATIONS)',
                  'UK BUS (LONDON)', 'UK OFFSHORE', 'UK ONSHORE', 'UK OPERATIONS',
                  'THAMES VALLEY', 'GREATER LONDON', 'UNITED KINDOM', 'REST OF UK',
                  'UK BUSINESS', 'UK- GROUPHAMESAFE', 'UK- ERA', 'UNITED KIGDOM',
                  'LONDON & SOUTH EAST', 'CENTRAL & SOUTH WEST', 'LONDON & SOUTH',
                  'CENTRAL & WEST', 'CENTRAL LONDON', 'U.K. EXPORT SHIPPERS', 'UK SHIPPERS',
                  'UK & CHANNEL ISLAND', 'UNITED KINGSOM', 'UNITETED KINGDOM', 'UK- EUROPE',
                  'WALES', 'UNITED KINDGOM', 'UNITED KINGDAM', 'UNITED K INGDOM',
                  'U.K EUROPEAN OPERATIONS', 'EUROPE (UK)', 'LONDON MARKET',
                  'FOREIGN (PRINCIPALLY U.K.)', 'UK DIVISION', 'BRITISH VRIGIN ISLAND',
                  'SCOTLAND/ENGLAND', 'BRITISB ISLANDS', 'UK HEAD OFFICE',
                  'UK CHANNEL ISLANDS', 'ALDERNEY', 'GILBRALTAR', 'ENGLAND/UK', 'BRITISH',
                  'SHETLAND -UK', 'BRITIAN', 'UK RETAIL', 'UK HERON',
                  'EUROPE(U.K)', 'CHANNEL ISLANDS AND THE UK', 'U K',
                  'UK (DOMICILE)', 'EUROPE (PRIMARILY U.K.)', 'EUROPE PRIMARILY UK',
                  'UK REGION', 'INTERNATIONAL(UK)', 'UK NORTH SEA'
                  # Not so precise
                  #,'UK AND EUROPE', 'UK & EUROPE', 'UK/EUROPE',  'GREATE BRITIAN & OTHER',
                  #'UK/IRLAND', 'UK & EIRE', 'THE UK AND EIRE', 'ENGLAND & OTHER',
                  #'UK & INTERNATIONAL', 'U.K. AND ELIMINATION', 'UK & OTHER REGIONS',
                  #'UK & ROI', 'FOREIGN (PRINCIPALLY U.K. & EUROPE)', 'UK & OTHER EUROPE',
                  #'EUROPE UNION/UK', 'UK AND OTHERS', 'UK AND EIRE','UK AND REST OF WORLD',
                  #'UK AND CENTRAL EUROPE', 'ALDERNEY & OTHER', 'UK AND OTHER',
                  #'ENGLAND (FIRST TIER DIVISION) & OTHER', 'UK & REST OF THE WORLD',
                  #'UK & REST OF THE WORL', 'UK &  EUROPE', 'UK & OTHER', 'EUROPE - UK',
                  #'U.K./ EUROPE', 'UK AND WORLD', 'U.K. AND EUROPE', 'U.K. AND OTHERS',
                  #'UK/I', 'UK & IE', 'U.K & OTHER'
                  ]
            countries['Ukraine'] = ['UKRINE', 'UKARINE', 'UNKRAINE']
            countries['United States'] = ['WASHINGTON -USA', 'UNITED SATES OF AMERICA', 'TEXAS',
                   'NEW ENGLAND', 'THE USA', 'THE U.S.A.',
                   'ALASKA', 'DELAWARE',
                   'EGIDE USA INC.', 'USA (INSURANCE ONLY)', 'THE U.S.A', 'U.S',
                   'UNITES STATES OF AMERICA', 'THE ISLAND OF SAIPAN', 'UNITES STATES',
                   'NEW YORK', 'HAWAII', 'PAGO', 'UNITE STATES', 'AMERICA (US)',
                   'CALIFORNIA', 'UNITED STATED OF AMERICA', 'U S A', 'UNITED STATE',
                   'COMMERCIAL PROP. DEV.-US', 'INFRASTRUCTURE DEVELOPMENT-US',
                   'AVINS USA', 'USA REGION', 'UNITED STATE OF AMERICA',
                   'UNITED STATED', 'US OPERATIONS',
                   'OTHERS MAINLY USA', 'OTHERS-MAINLY U.S.A.', 'OTHER MAINLY U.S.A.',
                   'USA JOINT VENTURES', 'USA JOINT VENTURE', 'US- AMESBURY',
                   'US- AMESBURYTRUTH', 'PORTSMOUTH', 'U.S.A. (NORTH AMERICA)',
                   'NORTH AMERICA - USA', 'UNITED STAES', 'UNITED SALES',
                   'PHILADELPHIA CBD', 'PENNSYLVANIA SUBURBS', 'METROPOLITAN WASHINGTON, D.C.',
                   'NEW JERSEY/DELAWARE', 'RICHMOND VIRGINIA', 'AUSTIN, TEXAS', 'AUSTIN/TEXAS',
                   'URBAN/METROPOLITAN D.C', 'AMERICAS (PRINCIPALLY THE U.S.)',
                   'UNITTED STATES', 'U.S. OPERATIONS', 'US TERRITORIES & FOREIGN',
                   'SOUTHERN CALIFORNIA', 'NORTHERN CALIFORNIA', 'PACIFIC NORTHWEST',
                   'SEATTLE METRO', 'UNITED STARES', 'UNIITED STATES', 'UNIRED STATES',
                   'UNITED  STATES', 'STAMFORD / NEW YORK', 'CHICAGO', 'UNIED STATES',
                   'U. S. MEDICAL', 'U. S.', 'U.S. & POSSESSIONS', 'CALIFORNIA (STATE)',
                   'MIAMI', 'U.S. MAINLAND', 'DOMESTIC- US', 'THE UNITED  STATES',
                   'THE UNITED STATE', 'LAS VEGAS OPERATIONS', 'WYNN BOSTON HARBOR',
                   'THE US', 'NORTHERN VIRGINI', 'MARYLAND', 'WASHINGTON (D.C)',
                   'SOUTHERN VIRGINIA', 'TEXAS (STATE)', 'OKLAHOMA', 'ROCKY MOUNTAINS',
                   'USA PRODUCTION', 'USA EXPLORATION', 'USA DIRECT', 'BOSTON',
                   'LOS ANGELES', 'MINNEAPOLIS', 'DENVER', 'SAN DIEGO', 'ATLANTA',
                   'DALLAS', 'HOUSTON', 'DETROIT', 'SAN FRANCISCO BAY', 'U.S.GOVERNMENT',
                   'NORTH AMERICA(PRIMARILY U.S)', 'NORTH AMERICA (PRIMARILY THE US)',
                   'UINTED STATES OF AMER', 'NEVADA', 'UITED STATES', 'WESTERN US',
                   'UINTED STATES', 'PORTLAND', 'U.S OPERATIONS', 'HAILE (US)', 'UNITE STARES',
                   'NEW YORK (STATE)', 'OREGON', 'NORTHERN AMERICA (MAINLY US)', 'U S',
                   'WASHINGTON D.C. AREA', 'NORTH CALIFORNIA', 'SOUTHLAND / LOS ANGEL',
                   'SAN DIEGO / RIVERSIDE', 'CENTRAL AND EASTERN U.S.', 'US SALES',
                   'TOTAL USA', 'UINITED STATES', 'NEVADA (STATE)', 'UNIED STATES OF AMERICA',
                   'DOMESTIC(US)', 'NEW JERSEY', 'FLORIDA', 'INDIANA', 'MASSACHUSETTS',
                   'ARIZONA', 'MALIBU U.S.', 'U.S. REVENUE', 'U.S. DOMESTIC',
                   'NORTH CAROLINA', 'PENNSYLVANIA', 'VIRGINIA/TENNESSEE',
                   'OHIO/RHODE ISLAND', 'UNITATED STATES', 'COLORADO', 'WASHINGTON',
                   'AMERICA -U.S', 'CALPIAN, INC. - U.S', 'CALPIAN COMMERCE - U.S.',
                   'TENNESSEE', 'UNITED  STATE', 'MINNESOTA', 'U.S.OPERATIONS',
                   'SOUTH CAROLINA',
                   'US DOMESTIC REVENUE'
                   # Not so precise
                   #,'USA AND OTHER', 'USA & OTHER', 'U.S.A. & OTHER COUNTRIES',
                   #'USA AND OTHERS', 'USA/OTHER', 'U.S. AND THE AMERICAS',
                   #'U.S.A. & OTHER', 'USA & REST OF THE WORLD',
                   #'USA & REST OF WORLD', 'USA AND OTHER COUNTRIES', 'NORTH AMERICA/USA',
                   #'UNITED STAES OTHERS', 'USA & OTHERS', 'EASTERN US & OTHER',
                   #'USA & INTERNATIONAL', 'U.S. & OTHER', 'U.S. AND OTHER'
                   ]
            countries['Venezuela'] = ['VENEZEULA', 'VENIZUELA']
            countries['Vietnam'] = ['VIETAM', 'VIENAM', 'VITENAM', 'VEITNAM', 'VINETNAM', 'VIATNAM',
                       'HO CHI MINH, THANH PHO SAI GON', 'BA RIA - VUNG TAU',
                       'BINH DUONG', 'HA NOI, THU DO', 'HO CHI MINH, THANH PHO [SAI GON]',
                       'CAT LAI PORT HO CHI MINH, THANH PHO SAI GON',
                       'HAI AN PORT HAI PHONG, THANH PHO', 'DAC LAC AND PHU YEN',
                       'QUANG NAM', 'HAI PHONG, THANH PHO', 'HUNG YEN', 'BAC LIEU',
                       'HO CHI MINH CITY', 'DA NANG, THANH PHO', 'DAKLAK',
                       'PHU YEN', 'HAI PHONG', 'YEN BAI', 'HOA BINH', 'HA GIANG',
                       'THAI BINH', 'BAC NINH', 'PHU THO', 'HA NOI', 'NOTHERN VIETNAM',
                       'SOUTHERN VIETNAM', 'NORTHERN VIETNAM', 'BINH DINH', 'DA NANG',
                       'DONG NAI', 'HO CHI MINH',
                       'HA NOI, THU DO /HO CHI MINH, THANH PHO [SAI GON] (',
                       'HA NOI, THU DO /HO CHI MINH, THANH PHO SAI GON (PR',
                       'GIA LAI', 'LAM DONG', 'VINH PHUC', 'THANH HOA',
                       'KHANH HOA', 'VINH LONG', 'HAI DUONG', 'HO CHI MINH OFFICE',
                       'DA NANG OFFICE', 'CAM RANH OFFICE', 'SAI GON CAM RANH COMPANY',
                       'SAI GON - CAM RANH COMPANY', 'CA MAU', 'AN GIANG']
            countries['Yugoslavia'] = ['YUGOSLAVIA', 'FORMER YUGOSLAVIAN COUNTRIES',
                          'FORMER YUGOSLAVIAN', 'YUGOSLAV REPUBLIC']
            countries['Zimbabwe'] = ['ZIMBAWE']

        # Complete  standardized names
        for c, ns in countries.items():
            des.loc[des.segment_description.isin(ns), 'standardized'] = c

        # Add ISO3 codes
        des['country_in']  = coco.convert(names=des.standardized, to='ISO3', not_found=np.nan)

        # Clean multiple countries matches
        for i, row in des.iterrows():
            if type(row['country_in']) == list:
                des.loc[i, 'country_in'] = np.nan

        # Adjustment for outside territories

        # Identify domestic countries - TODO
        #des3 = des[des.standardized=='not found']
        #des3.to_csv('descripions_domestic_todo.csv', index=False)

        domestic = ['DOMESTIC', 'DOMESTIC MARKET','LOCAL',
                    'DOMESTIC MARKE', 'DOMESTIC REGION',
                    'DOMESTIC PORTION',
                    'DOMESTIC OPERATIONS',
                    'DOMESTIC SEGMENT', 'DOMESTIC OPERATION',
                    'DOMESTIC BRANCHES AND OFFICES', 'DOMESTIC BANKING UNIT',
                    'DOMESTIC ACTIVITIES',
                    'DOMESTC',
                    'DOMESTICS', 'DOMESTIC REINSURANCE INWARD BUSINESS',
                    'DOMESTIC MARKETS',
                    'TOTAL DOMESTIC OPERATIONS',
                    'PETROLEUM PRODUCTS - DOMESTIC',
                    'GAS - DOMESTIC', 'PETROCHEMICAL PRODUCTS - DOMESTIC',
                    'CRUDE OIL - DOMESTIC', 'DOMESTIC (UAE)',
                    'DOMESTICE', 'DOMESTIC SERVICES',
                    'DOMESTI MARKET', 'TOTAL DOMESTIC',
                    'DOMESTIC-HUADONG', 'DOMESTIC-OTHER REGION'
                    # Not so precise
                    #, 'DOMESTIC SALES', 'DOMESTIC SALE',
                    #'DOMESTIS SALES', 'DOMESTIC REVENUES',
                    #'SALES TO DOMESTIC MARKET', 'DOMESTIC REVENUE',
                    #'DOMESTIC REVENUES-TRANSPORTATION',
                    #'DOMESTIC REVENUES-BUSINESS UNIT',
                    #'DOMESTIC REVENUES-OTHERS',
                    #'DOMESTIC SALES & SERVICES', 'DOMESTIC SALES (INCL PROP DEV)',
                    #'OTHER DOMESTIC', 'DOMESTIC EXPORT SALES',
                    #'DOMESTIC STREAMING AND DVD', 'DOMESTIC STREAMING & DVD',
                    #'DOMESTIC DVD', 'DOMESTIC TARIFF',
                    #'DOMESIC SALES', 'ALL DOMESTIC REVENUE',
                    #'DOMESTIC TURNOVER', 'DOMESTIC DEMAND'
                    #'SALES OF PRODUCTS AND SERVICES/DOMESTIC MARKET',
                    #'INCOME FROM SALES/DOMESTIC MARKET',
                    #'SALES OF SERVICES AND PRODUCTS/DOMESTIC MARKET',
                    #'SALES OF GOODS/DOMESTIC MARKET',
                    #'SALES/DOMESTIC MARKET', 'SALES OF PRODUCTS AND SERVICES/DOMESTIC',
                    #'SALES OF GOODS/DOMESTIC', 'SALES ON DOMESTIC MARKET',
                    #'DOT DOMESTIC', 'LOCATION OF DOMESTIC',
                    #'DOMESTIC FREIGHT CHARGES',
                    ]

        des.loc[des.segment_description.isin(domestic), 'country_in'] = 'DOMESTIC'

        ############################################
        # Merge countries (IN and OUT) to the data #
        ############################################

        # Merge the ISO code to the segments
        dfin = dflc.reset_index().merge(des[['segment_description', 'country_in']], how='left')
        # dfin = dfin.merge(nat[['nation', 'country_out']], how='left')
        # Set domestic to country_out
        dfin.loc[dfin.country_in=='DOMESTIC', 'country_in'] = dfin.loc[dfin.country_in=='DOMESTIC', 'country_out']

        ########################
        # Clean Financial data #
        ########################
        # Ensure (firm, country) uniqueness
        # May need to aggregate measures as some geographic segments
        # are finer than a country
        key = self.key + ['country_in', 'country_out']
        cols = ['segment_sales', 'segment_oic', 'segment_assets',
                'segment_capex', 'segment_depreciation']
        #cols_add = ['sic1', 'sic2', 'sic3', 'sic4',
        #            'sic5', 'sic6', 'sic7', 'sic8',
        #            'cusip', 'isin', 'sedol', 'ticker']
        # Used to set NA values when no data
        dfin_count = dfin.groupby(key)[cols].count().reset_index()
        dfin = dfin.groupby(key)[cols].sum().reset_index()
        dfin[dfin_count==0] = np.nan
        # Add other info
        #dfin_other = dfin[key+cols_add].dropna(subset=key).drop_duplicates()
        #dfin3 = dfin2.merge(dfin_other, on=key, how='left')
        return(dfin)




    #################
    # New Variables #
    #################

    def _isonation(self, data):
        """ Return the ISO3 code corresponding to item6026 (nation)
        """
        key = self.key
        df = self.open_data(data, key)
        fields = ['item6026']
        df['nation'] = self.get_fields(fields, data)
        # Extract countries info
        df.nation = df.nation.str.upper()
        nat = pd.DataFrame(df.reset_index().nation.drop_duplicates().reset_index(drop=True))
        ## Standardize names
        nat['standardized'] = coco.convert(names=nat.nation, to='name_short')
        # Hand cleaning of not identified countries
        nat.loc[nat.nation.isin(['CHANNEL ISLANDS', 'ENGLAND']), 'standardized'] = 'United Kingdom'
        # Add ISO3 codes
        nat['iso']  = coco.convert(names=nat.standardized, to='ISO3', not_found=np.nan)
        # Merge
        df = df.merge(nat[['nation', 'iso']], how='left')
        # Return the ISO code
        df.index = data.index
        return(df.iso)
