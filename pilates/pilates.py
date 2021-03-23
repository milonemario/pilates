"""
Python module to create data for Accounting, Finance, and Economic Research.

It implements general procedures to process financial and accounting
data from WRDS and other useful sources.

The module implements several classes to process different data sources, such
as Compustat, CRSP, IBES, FRED, etc.
"""

# import sys
import os, sys, getpass, stat
import wget, tarfile, gzip, zipfile
import pathlib
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp
import importlib
import psycopg2
from openpyxl import load_workbook
# import dask.dataframe as dd

# Remove warning messages for chained assignments
pd.options.mode.chained_assignment = None

# Directories
_types_dir = os.path.dirname(__file__)+'/types/'
_modules_dir = os.path.dirname(__file__)+'/modules/'

# WRDS Database Connection
WRDS_POSTGRES_HOST = 'wrds-pgdata.wharton.upenn.edu'
WRDS_POSTGRES_PORT = 9737
WRDS_POSTGRES_DB = 'wrds'

####################
# Global functions #
####################

def check_duplicates(df, key, description=''):
    """ Check duplicates and print a message if needed.

    Args:
        df (pandas.DataFrame): DataFrame.
        key (list): Columns that should disp;lay no duplicates.
        description (str, optional): String to include in the warning message

    """
    n_dup = df.shape[0] - df[key].drop_duplicates().shape[0]
    if n_dup > 0:
        print("Warning: The data {:} contains {:} \
              duplicates".format(description, n_dup))

def __correct_columns_types(df, types=None):
    """ Apply the correct data type to all known columns.
    Known columns are listed in the files contained in the folder 'types'.
    A custom type file can be provided by the user.

    Depreciated.

    Args:
        df (pandas.DataFrame): DataFrame.
        types (str, optional): Path of the Yaml file containing the fields
            types of the data.

    """

    def get_changes(path):
        with open(path) as f:
            t = yaml.full_load(f)
            # Create the final type list
            types_list1 = {}  # Convert to float before Int64
            types_list2 = {}
            for k, l in t.items():
                for v in l:
                    types_list1[v] = k
                    types_list2[v] = k
                    if k in ['Int64', 'Int32']:
                        types_list1[v] = 'float'
        # Select the common columns
        cols_df = df.columns
        cols_change = [v for v in cols_df if v in types_list2.keys()]
        types_list_ch1 = {k: types_list1[k] for k in cols_change}
        types_list_ch2 = {k: types_list2[k] for k in cols_change
                          if types_list2[k] in ['Int64', 'Int32']}
        return([types_list_ch1, types_list_ch2])

    def apply_changes(ch):
        # Apply the date and non-date types separately
        chd = {k: v for k, v in ch.items() if v == 'date'}
        chnd = {k: v for k, v in ch.items() if v != 'date'}
        # Apply the non-dates
        if len(chnd) > 0:
            df.loc[:, chnd.keys()] = df[chnd.keys()].astype(chnd)
        # Apply the dates
        if len(chd) > 0:
            for k in chd.keys():
                df.loc[:, k] = pd.to_datetime(df[k]).dt.date
                df.loc[df[k].isna(), k] = np.nan

    # First use the predefined types
    for fname in os.listdir(_types_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.yaml':
            path = _types_dir+fname
            changes = get_changes(path)
            for ch in changes:
                apply_changes(ch)
    # Then use the user provided types
    if types is not None:
        changes = get_changes(types)
        for ch in changes:
            apply_changes(ch)
    return(df)

##############
# Main Class #
##############

class data:
    """ Main class of pilates.

    Args:
        datadir (str): Path of the directory used to store the data files
            for use by the module.
        chunksize (int): Size of chunk to use when converting files and
            downloading SQL requests.

    """

    def __init__(self, datadir=None, chunksize=500000):
        self.datadir = None
        if datadir is not None:
            self.set_data_directory(datadir)
        self.chunksize = chunksize
        self.cores = mp.cpu_count()
        # Add the different modules
        for m in os.listdir(_modules_dir):
            for f in os.listdir(_modules_dir+m):
                name, ext = os.path.splitext(f)
                if ext == '.py':
                    mod = importlib.__import__('pilates.modules.'+m+'.'+name,
                                               fromlist=[name])
                    class_ = getattr(mod, name)
                    instance = class_(self)
                    setattr(self, name, instance)

    def _check_data_dir(self):
        """ Check if the data directory has been specified.

        Raises:
            Exception: If the data directory has not been set

        """
        if self.datadir is None:
            raise Exception("Please provide a data directory with",
                            "'set_data_directory()'")

    def set_chunksize(self, size):
        """ Defines the chunk size to use when converting datasets.

        Args:
            Size (int): Chunk size.

        """
        self.chunksize = size

    def set_data_directory(self, datadir):
        """ Defines the data directory to store the data files.

        Args:
            datadir (str): Path of the directory

        """
        if datadir[-1] != '/':
            self.datadir = datadir+'/'
        else:
            self.datadir = datadir
        # Set directory for raw files downloads
        self.datadownload = datadir + 'downloads/'
        # Create the directories if it does not exist
        pathlib.Path(self.datadir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.datadownload).mkdir(parents=True, exist_ok=True)

    def set_data_frequency(self, frequency):
        """ Define the frequency of the user's data.

        Args:
            frequency (str): Frequency of the data.
                Supports 'Daily', 'Monthly', 'Quarterly' and 'Annual'.

        Raises:
            Exception: If the frequency is not supported.

        """
        if frequency in ['Daily', 'daily', 'D', 'd']:
            self.freq = 'D'
        elif frequency in ['Monthly', 'monthly', 'M', 'm']:
            self.freq = 'M'
        elif frequency in ['Quarterly', 'quarterly', 'Q', 'q']:
            self.freq = 'Q'
        elif frequency in ['Annual', 'annual', 'A', 'a',
                           'Yearly', 'yearly', 'Y', 'y']:
            self.freq = 'A'
        else:
            raise Exception('The data frequency must be either' +
                            ' Daily, Monthly, Quarterly, or Annual')

    def set_date_column(self, col_date, type=None):
        """ Define the date column to use for the user's data.

        Args:
            col_date (str): name of the columns to be used as the date column.
            type:   Type of the date column ('date', 'year')
                    By default, consider the date to be at the daily level.
                    If given type='year', considers that the date is only a year.

        """
        self.col_date = col_date
        if type in ['date', 'year']:
            self.col_date_type = type
        elif type is None:
            self.col_date_type = 'date'
        else:
            raise Exception("The type of date must be in ['date', 'year'].")

    def set_cores_no(self, cores):
        """ Set the number of cores to use for parallel computations.

        Args:
            cores (int): Number of cores to use.
        """
        self.cores = cores

#############################
# General class for modules #
#############################

class data_module:
    """ Class inherited by all the modules used to process the data.

    Args:
        w (pilates.data): Instance of the main class.

    Attributes:
        col_id (str): Name of the column used as identifier.
        cl_date (str): Name of the column used as date.

    """

    def __init__(self, d):
        # Provide access to the main data class
        self.d = d
        # By default, modules use local files
        self.remote_access = False
        # Get informations on files supported by the module
        path_files = _modules_dir + self.__class__.__name__ + '/files.yaml'
        if os.path.exists(path_files):
            with open(path_files) as f:
                self.files = yaml.full_load(f)
        else:
            self.files = None
        # Get information on fields types
        path_types = _modules_dir + self.__class__.__name__ + '/types.yaml'
        if os.path.exists(path_types):
            with open(path_types) as f:
                self.types= yaml.full_load(f)
        else:
            self.types = None

    ###########################################################
    # Helper functions for handling file conversion and other #
    ###########################################################

    def _filepath_ext(self, name):
        """ Return the full path (with extenssion) of the file to dowload.
        The file can be  a compressed file.
        """
        #url = self.files[name]['url']
        filename = self.files[name]['file']
        filepath_ext = self.d.datadownload + filename
        return filepath_ext

    def _path(self, name):
        """ Return the full path (with extension) of the file to be converted.
        The file is an uncompressed file
        """
        if 'file' in self.files[name].keys():
            path = self.d.datadownload + self.files[name]['file']
        else:
            url = self.files[name]['url']
            filename, ext = os.path.splitext(os.path.basename(url))
            filetype = self.files[name]['type']
            path = self.d.datadownload + filename + '.' + filetype
        return path

    def _filepath_pq(self, name):
        """ Return the full path of the parquet file to be used by the module.
        """
        if name in self.files.keys():
            filename = self.__class__.__name__ + '_' + name
        else:
            filename = name
        filename_pq = self.d.datadir + filename + '.parquet'
        return filename_pq

    def _check_name(self, name):
        """ Check that the file name is allowed for that module.
        For example, the module for COMPUSTAT (comp) only handles certain files
        such as 'funda', 'fundq', 'names'.

        Args:
            name (str): Name of the file to be used by the module.
        """
        if self.files:
            if name not in self.files.keys():
                raise Exception("The file name provided is incorrect. "
                                "Module "+self.__class__.__name__+"supports "
                                "the following files: "+str(list(self.files.keys())))

    def _write_to_parquet(self, df, name):
        filepath_pq = self._filepath_pq(name)
        df = df[df.columns.dropna()]
        df.columns = map(str.lower, df.columns)  # Lower case col names
        df = self._correct_columns_types(df)
        t = pa.Table.from_pandas(df)
        pqwriter = pq.ParquetWriter(filepath_pq, t.schema)
        pqwriter.write_table(t)
        pqwriter.close()

    def _convert_data_by_chunks(self, f, filepath_pq, nrows):
        """ Convert the file by chunks

        Args:
            f (iterator): file already opened by chunks.
            filepath_pq (str): Path of the parquet file to create.
            nrows (int): Number of rows in the file.

        """
        # Write the data
        pqwriter = None
        # pqschema = None
        for i, df in enumerate(f):
            df = self._process_fields(df)
            df.columns = map(str.lower, df.columns)  # Lower case col names
            df = self._correct_columns_types(df)
            nobs = (i+1)*f.chunksize
            print("Progress conversion : {:2.0%}".format(nobs/float(nrows)), end='\r')
            if i == 0:
                # Correct the column types on the first chunk to get
                # the correct schema
                t = pa.Table.from_pandas(df)
                pqschema = t.schema
                pqwriter = pq.ParquetWriter(filepath_pq, t.schema)
            else:
                t = pa.Table.from_pandas(df, schema=pqschema)
                # t = pa.Table.from_pandas(df)
            pqwriter.write_table(t)
        print('\r')
        pqwriter.close()

    def _convert_data(self, name, path,
                     force=False,
                     delim_whitespace=False,
                     nrows=None):
        """ Convert the file to parquet and save it in the
        data directory.

        This method currently support the following file formats:
            - CSV
            - SAS (.sas7bdat)

        Args:
            path (str): Path of the file to convert.
            force (bool, optional): If False, the file is not re-converted.
                Defaults to False.
            delim_whitespace (bool): Set to True if the file is a CSV with
                baln separateors (tab for instance). Passed to pandas.read_csv().
                Defaults to False.

        Todo:
            * Add support for more file formats (Stata, ...)

        """
        self.d._check_data_dir()
        # Create the new file name
        filepath_pq = self._filepath_pq(name)
        # Check if the file has already been converted
        if not os.path.exists(filepath_pq) or force:
            print('Conversion '+name)
            # Get the file type
            filename, ext = os.path.splitext(os.path.basename(path))
            # Ovewride the file type if provided
            if 'type' in self.files[name]:
                ext = '.' + self.files[name]['type']
            # Open the file (by chunks)
            if ext == '.sas7bdat':
                f = pd.read_sas(path, chunksize=self.d.chunksize)
                # Get the total number of rows
                nrows = f.row_count
                # Convert the file
                self._convert_data_by_chunks(f, filepath_pq, nrows)
            elif ext in ['.csv', '.asc']:
                f = pd.read_csv(path, chunksize=self.d.chunksize,
                                delim_whitespace=delim_whitespace)
                # Get the total number of rows
                # Need to open the file (only one column)
                f_tmp = pd.read_csv(path, usecols=[0],
                                    delim_whitespace=delim_whitespace)
                nrows = f_tmp.shape[0]
                del(f_tmp)
                self._convert_data_by_chunks(f, filepath_pq, nrows)
            elif ext in ['.xls']:
                # Open the file
                sheet_name = self.files[name]['sheet']
                df = pd.read_excel(path, sheet_name=sheet_name, nrows=nrows)
                # For Excel files, writes the whole file directly
                self._write_to_parquet(df, name)
            elif ext in ['.xlsx']:
                wb = load_workbook(path)
                ws = wb[self.files[name]['sheet']]
                data = ws.values
                cols = next(data)
                if nrows:
                    data = list(data)[0:nrows-1]
                else:
                    data = list(data)
                df = pd.DataFrame(data, columns=cols)
                # For Excel files, writes the whole file directly
                self._write_to_parquet(df, name)
            else:
                raise Exception("This file format is not currently "
                                "supported. Supported formats are: "
                                ".sas7bdat, .csv")

    def _process_fields(self, df):
        """ Properly encode the string fields (remove bytes string types).

        Args:
            df (pandas.DataFrame): DataFrame for which to process the string
                fields.

        """
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].where(df[c].apply(type) != bytes,
                                    df[c].str.decode('utf-8', errors='ignore'))
        return df

    def _open_local_data(self, name, columns=None):
        """ Open the data.
        Only return data from locally stored files.

        Args:
            name (str or pandas.DataFrame):
                Either the name of the file to open or a DataFrame.
                When a name is given, the file must have been converted using
                convert_data().
            columns (list, optional): Columns to keep. If None, returns all
                the columns. Defaults to None.
            types (str, optional): Path of the Yaml file containing the fields
                types of the data.

        Returns:
            pandas.DataFrame: DataFrame with the required columns.

        """
        if isinstance(name, pd.DataFrame):
            # If the name refers to a pandas DataFrame, just return it
            df = name[columns]
        else:
            self.d._check_data_dir()
            filepath_pq = self._filepath_pq(name)
            t = pq.read_table(filepath_pq, columns=columns)
            df = t.to_pandas()
            del(t)
        return df

    def _get_local_fields_names(self, name):
        """ Get the fields names of the file.
        Only returns fields from locally stored files.

        Args:
            name (str or pandas.DataFrame):
                Either the path of the parquet file to open or a DataFrame.

        Returns:
            list: List containing the fields names.

        """
        if isinstance(name, pd.DataFrame):
            cols = name.columns
        else:
            # Otherwise, open the file from disk
            self.d._check_data_dir()
            # Open the parquet file and convert it to a pandas DataFrame
            filepath_pq = self._filepath_pq(name)
            schema = pq.read_schema(filepath_pq)
            cols = schema.names
        return(cols)

    def _correct_columns_types(self, df):
        """ Apply the correct data type to all known columns.
        Known columns are listed in the files contained in the folder 'types'.
        A custom type file can be provided by the user.

        Args:
            df (pandas.DataFrame): DataFrame.
        """
        def get_changes():
            # Create the final type list
            types_list1 = {}  # Convert to float before Int64
            types_list2 = {}
            for k, l in self.types.items():
                for v in l:
                    types_list1[v] = k
                    types_list2[v] = k
                    if k in ['Int64', 'Int32']:
                        types_list1[v] = 'float'
            # Select the common columns
            cols_df = df.columns
            cols_change = [v for v in cols_df if v in types_list2.keys()]
            types_list_ch1 = {k: types_list1[k] for k in cols_change}
            types_list_ch2 = {k: types_list2[k] for k in cols_change
                              if types_list2[k] in ['Int64', 'Int32']}
            return([types_list_ch1, types_list_ch2])

        def apply_changes(ch):
            # Apply the date and non-date types separately
            chd = {k: v for k, v in ch.items() if v == 'date'}
            chnd = {k: v for k, v in ch.items() if v != 'date'}
            # Apply the non-dates
            if len(chnd) > 0:
                for k in chnd.keys():
                    df.loc[:, k] = df[k].astype(chnd[k])
            # Apply the dates
            if len(chd) > 0:
                for k in chd.keys():
                    #df.loc[:, k] = pd.to_datetime(df[k]).dt.date
                    df.loc[:, k] = pd.to_datetime(df[k])
                    #df.loc[df[k].isna(), k] = np.nan

        # Then use the user provided types
        if self.types:
            changes = get_changes()
            for ch in changes:
                apply_changes(ch)
        return(df)

    #################
    # Use functions #
    #################

    def set_remote_access(self, remote_access=True):
        self.remote_access = remote_access

    def download_file(self, name):
        url = self.files[name]['url']
        filepath_ext = self._filepath_ext(name)
        _, ext = os.path.splitext(os.path.basename(url))
        path = self._path(name)
        if not os.path.exists(path):
            # Download the file
            if not os.path.exists(path):
                print('Download file '+name+' for module '+self.__class__.__name__+' ...')
                wget.download(url, self.d.datadownload)
                print('\n')
            # Uncompress file or packages if needed
            if ext == '.gz':
                # Here: only supports one compressed file
                f = gzip.open(filepath_ext, 'rt')
                content = f.read()
                with open(path, 'w') as fn:
                    fn.write(content)
            elif ext == '.zip':
                with zipfile.ZipFile(filepath_ext, 'r') as zip_ref:
                    zip_ref.extractall(self.d.datadownload)

    def add_file(self, name, path=None, force=False,
                 # For CSV files
                 delim_whitespace=False,
                 # for Excel files
                 nrows=None):
        """ Add a file to the module.
        Converts and make the file available for the module to use.

        Args:
            name (str): Name of the file to be used by the module.
            path (str): Path of the original file to convert.
            force (bool, optional): If False, the file is not re-converted.
                Defaults to False.
            types (str, optional): Path of the Yaml file containing the fields
                types of the data.

        """
        # Check the file name (if it is part of the allowed files)
        self._check_name(name)
        if not hasattr(self, name):
            if self.remote_access and not path:
                # Module is requested the fetch the file by itself
                url = self.files[name]['url']
                # Get extension
                filepath_ext = self._filepath_ext(name)
                path = self._path(name)
                filepath_pq = self._filepath_pq(name)
                if not os.path.exists(filepath_pq):
                    self.download_file(name)
                    # Set arguments
                    if 'delim_whitespace' in self.files[name].keys():
                        delim_whitespace = self.files[name]['delim_whitespace']
                    if 'nrows' in self.files[name].keys():
                        nrows = self.files[name]['nrows']
            elif not path:
                raise Exception('The module is not set to getch remote files '
                                'and no file path is provided.')
            # Convert the file
            f = self._convert_data(name, path, force=force,
                                    delim_whitespace=delim_whitespace,
                                    nrows=nrows)
            # Add the file to the module
            setattr(self, name, name)

    def open_data(self, name, columns=None):
        """ Open the data.
        If the module is set to use remote files, update local files accordingly.
        Otherwise, use local files.

        Args:
            name (str or pandas.DataFrame):
                Either the name of the file to open or a DataFrame.
                When a name is given, the file must have been converted using
                convert_data().
            columns (list, optional): Columns to keep. If None, returns all
                the columns. Defaults to None.
            types (str, optional): Path of the Yaml file containing the fields
                types of the data.

        Returns:
            pandas.DataFrame: DataFrame with the required columns.

        """

        if (self.remote_access and
           not isinstance(name, pd.DataFrame) and
           columns is not None):
            # Update or create the file on disk
            missing_fields = columns.copy()
            # Check that a data directory has been provided
            self.d._check_data_dir()
            filepath_pq = self._filepath_pq(name)
            # Check if there is a file on disk to be used
            if os.path.exists(filepath_pq):
                # If the file exists, update the missing fields
                local_fields = self._get_local_fields_names(name)
                for c in columns:
                    if c in local_fields:
                        missing_fields.remove(c)
            # Update or create the file on disk
            if len(missing_fields) > 0:
                self._add_fields_to_file(name, missing_fields)
        # The file on disk should be good to use now.
        return self._open_local_data(name, columns)

    def get_fields_names(self, name):
        """ Get the fields names of the file.
        If module is set to use remote files, get all remote fields.
        Otherwise return fields from local files.

        Args:
            name (str or pandas.DataFrame):
                Either the path of the parquet file to open or a DataFrame.

        Returns:
            list: List containing the fields names.

        """

        if self.remote_access:
            return self._get_remote_fields_names(name)
        else:
            return self.d._get_local_fields_names(name)

    def get_lag(self, data, lag, fields=None, col_id=None, col_date=None):
        """ Return the lag of the columns in the data (or the fields if
        specified).

        Arguments:
            data (pd.DataFrame): DataFrame
            lag (int):  Number of lags. If negative, returns future values.
            fields (list, optional): If specified, only return the lag of
                these fields. Default to None (all fields).
            col_id (str): Identifier column (gvkey, ticker, permno, ...)
                If None, use the module identifier column (self.col_id)
                Defaults to None.
            col_date: Date column (datadate, fpedats, date, ...)
                If None, use  the module date columns (self.col_date)
                Defaults to None.
        """
        if col_id is None:
            col_id = self.col_id
        if col_date is None:
            col_date = self.col_date
        all_fields = [f for f in data.columns if f not in [col_id, col_date]]
        if lag != 0:
            data = data.sort_values([col_id, col_date])
            data_l = data.groupby(col_id)[all_fields].shift(lag)
            data[all_fields] = data_l
        if fields is None:
            return(data)
        else:
            return(data[fields])

##########################
# Class for WRDS modules #
##########################

class wrds_module(data_module):
    """ Module class for all modules that use WRDS data.

    """
    # WRDS connections
    conn = {}

    def __init__(self, d):
        data_module.__init__(self, d)

    ####################
    # Helper functions #
    ####################

    def _get_remote_fields_names(self, name):
        """ Obtain the fields of the file 'name' from WRDS postgresql data.

        """
        schema = self.files[name]['schema']
        table = self.files[name]['table']
        sqlstmt = ('SELECT * FROM {schema}.{table} LIMIT 0;'.format(
                   schema=schema,
                   table=table))
        cursor = self.conn.cursor()
        cursor.execute(sqlstmt)
        cols = [desc[0] for desc in cursor.description]
        cursor.close()
        return cols

    def _add_fields_to_file(self, name, fields):
        # Get the missing fields
        schema = self.files[name]['schema']
        table = self.files[name]['table']
        index = self.files[name]['index']
        print('Downloading fields '+str(fields)+' for module',
              self.__class__.__name__+', table '+table+' from WRDS ... ')
        # Open and update existing file if any
        filepath_pq = self._filepath_pq(name)
        file_exists = False
        if os.path.exists(filepath_pq):
            file_exists = True
            pqf = pq.ParquetFile(filepath_pq)
        filepath_pq_new = filepath_pq + '_new'

        # Get approximate row count
        nrows = self.get_row_count(schema, table)
        # SQL query
        cols = ','.join(fields)
        order = ', '.join(index)
        sqlstmt = ('SELECT {cols} FROM {schema}.{table} ORDER BY {order};'.format(
                   cols=cols,
                   schema=schema,
                   table=table,
                   order=order))
        # Read the SQL table by chunks
        cursor = self.conn.cursor(name='mycursor')
        cursor.itersize = self.d.chunksize
        cursor.execute(sqlstmt)
        more_data = True
        i = 0
        while more_data:
            chunk = cursor.fetchmany(cursor.itersize)
            dfsql = pd.DataFrame(chunk, columns=fields)
            # Correct types
            dfsql.columns = map(str.lower, dfsql.columns)  # Lower case col names
            dfsql = self._correct_columns_types(dfsql)
            if file_exists:
                df = pqf.read_row_group(i).to_pandas()
                # Merge old and new data
                df[fields] = dfsql
            else:
                df = dfsql
            if i == 0:
                t = pa.Table.from_pandas(df)
                pqschema = t.schema
                pqwriter = pq.ParquetWriter(filepath_pq_new, t.schema)
            else:
                t = pa.Table.from_pandas(df, schema=pqschema)
            pqwriter.write_table(t)
            i += 1
            if len(df) < self.d.chunksize:
                more_data = False
                print('Progress: Done')
            else:
                nobs = (i+1)*self.d.chunksize
                print("Progress: {:2.0%}".format(nobs/float(nrows)), end='\r')
        pqwriter.close()
        cursor.close()
        # Remove old file and rename new file
        if file_exists:
            os.remove(filepath_pq)
        os.rename(filepath_pq_new, filepath_pq)

    ################
    # Overloadings #
    ################

    def set_remote_access(self, remote_access=True, wrds_username=None):
        data_module.set_remote_access(self, remote_access)
        # WRDS Library
        self.wrds_username = wrds_username
        if self.remote_access:
            # Check if already a connection from this WRDS user
            if self.wrds_username not in wrds_module.conn.keys():
                # First create a pgpass file for subsequent connections
                self.create_pgpass_file()
                print('Connecting user {} to WRDS database ... '.format(self.wrds_username), end='', flush=True)
                # self.connwrds = wrds.Connection(wrds_username = wrds_username)
                wrds_module.conn[self.wrds_username] = psycopg2.connect("host={} dbname={} user={} port={}".format(WRDS_POSTGRES_HOST,
                    WRDS_POSTGRES_DB, wrds_username, WRDS_POSTGRES_PORT))
                #self.views = self.__get_view_names();
                #self.tables = self.__get_table_names();
                print('established.')
                # Add all files supported by the module
                path_files = _modules_dir + self.__class__.__name__ + '/files.yaml'
                with open(path_files) as f:
                    names = yaml.full_load(f)
                    for name in names.keys():
                        setattr(self, name, name)
            # Make the connection available to the module
            self.conn = wrds_module.conn[self.wrds_username]

    ########################################
    # Replicate some wrds module functions #
    ########################################

    def __get_view_names(self):
        sqlcode = "select viewname from pg_catalog.pg_views;"
        with self.conn.cursor() as curs:
            curs.execute(sqlcode)
            views = curs.fetchall()
            return [v[0] for v in views]

    def __get_table_names(self):
        sqlcode = "select tablename from pg_catalog.pg_tables;"
        with self.conn.cursor() as curs:
            curs.execute(sqlcode)
            tables = curs.fetchall()
            return [t[0] for t in tables]

    def __get_schema_for_view(self, schema, table):
        """
        Internal function for getting the schema based on a view
        """
        sql_code = """SELECT distinct(source_ns.nspname) AS source_schema
                      FROM pg_depend
                      JOIN pg_rewrite
                        ON pg_depend.objid = pg_rewrite.oid
                      JOIN pg_class as dependent_view
                        ON pg_rewrite.ev_class = dependent_view.oid
                      JOIN pg_class as source_table
                        ON pg_depend.refobjid = source_table.oid
                      JOIN pg_attribute
                        ON pg_depend.refobjid = pg_attribute.attrelid
                          AND pg_depend.refobjsubid = pg_attribute.attnum
                      JOIN pg_namespace dependent_ns
                        ON dependent_ns.oid = dependent_view.relnamespace
                      JOIN pg_namespace source_ns
                        ON source_ns.oid = source_table.relnamespace
                      WHERE dependent_ns.nspname = '{schema}'
                        AND dependent_view.relname = '{view}';
                    """.format(schema=schema, view=table)
        with self.conn.cursor() as curs:
            curs = self.conn.cursor()
            curs.execute(sql_code)
            return curs.fetchone()[0]

    def get_row_count(self, schema, table):
        """
            Uses the library and table to get the approximate
              row count for the table.
            :param library: Postgres schema name.
            :param table: Postgres table name.
            :rtype: int
            Usage::
            >>> db.get_row_count('wrdssec', 'dforms')
            16378400
        """
        if 'taq' in schema:
            print("The row count will return 0 due to the structure of TAQ")
        #else:
        #    if table in self.views:
        #        schema = self.__get_schema_for_view(schema, table)
        #if schema:
        sqlstmt = """
            SELECT reltuples
              FROM pg_class r
              JOIN pg_namespace n
                ON (r.relnamespace = n.oid)
              WHERE r.relkind in ('r', 'f')
                AND n.nspname = '{}'
                AND r.relname = '{}';
            """.format(schema, table)

        try:
            with self.conn.cursor() as curs:
                curs.execute(sqlstmt)
                return int(curs.fetchone()[0])
        except Exception as e:
            print(
                "There was a problem with retrieving"
                "the row count: {}".format(e))
            return 0
        #else:
        #    print("There was a problem with retrieving the schema")
        #    return None

    def create_pgpass_file(self):
        """
        Create a .pgpass file to store WRDS connection credentials..
        """

        if not self.wrds_username:
            self.wrds_username = input("Enter your WRDS username:")
        if (sys.platform == 'win32'):
            pgfile = self.__pgpass_file_win32()
        else:
            pgfile = self.__pgpass_file_unix()
        if not self.__pgpass_exists(pgfile):
            self._wrds_passwd = getpass.getpass('Enter the WRDS password for {}:'.format(self.wrds_username))
            self.__write_pgpass_file(pgfile)

    def __pgpass_file_win32(self):
        """
        Create a pgpass.conf file on Windows.
        Windows is different enough from everything else
          as to require its own special way of doing things.
        Save the pgpass file in %APPDATA%\postgresql as 'pgpass.conf'.
        """
        appdata = os.getenv('APPDATA')
        pgdir = appdata + os.path.sep + 'postgresql'
        # Can we at least assume %APPDATA% always exists? I'm seriously asking.
        if (not os.path.exists(pgdir)):
            os.mkdir(pgdir)
        # Path exists, but is not a directory
        elif (not os.path.isdir(pgdir)):
            err = ("Cannot create directory {}: "
                   "path exists but is not a directory")
            raise FileExistsError(err.format(pgdir))
        pgfile = pgdir + os.path.sep + 'pgpass.conf'
        return pgfile

    def __pgpass_file_unix(self):
        """
        Create a .pgpass file on Unix-like operating systems.
        Permissions on this file must also be set on Unix-like systems.
        This function works on Mac OS X and Linux.
        It should work on Solaris too, but this is untested.
        """
        homedir = os.getenv('HOME')
        pgfile = homedir + os.path.sep + '.pgpass'
        return pgfile

    def __write_pgpass_file(self, pgfile):
        """
        Write the WRDS connection info to the pgpass file
          without clobbering other connection strings.
        Also escape any ':' characters in passwords,
          as .pgpass requires.
        Works on both *nix and Win32.
        """
        pgpass = "{host}:{port}:{dbname}:{user}:{passwd}"
        passwd = self._wrds_passwd
        passwd = passwd.replace(':', '\:')
        line = pgpass.format(
            host=WRDS_POSTGRES_HOST,
            port=WRDS_POSTGRES_PORT,
            dbname=WRDS_POSTGRES_DB,
            user=self.wrds_username,
            passwd=passwd)
        lines = [line]
        if (os.path.isfile(pgfile)):
            if not (sys.platform == 'win32'):
                # Set it to mode 600 (rw-------) so we can write to it
                os.chmod(pgfile, stat.S_IRUSR | stat.S_IWUSR)
            file = open(pgfile, 'a')
        else:
            file = open(pgfile, 'w')

        # Add a new line or create file
        file.writelines(lines)
        file.write('\n')
        file.close()
        if not (sys.platform == 'win32'):
            # Set it to mode 400 (r------) to protect it
            os.chmod(pgfile, stat.S_IRUSR)

    def __pgpass_exists(self, pgfile):
        if (os.path.isfile(pgfile)):
            with open(pgfile, 'r') as fd:
                lines = fd.readlines()
            for line in lines:
                oldline = line.replace("""\:""", '##COLON##')
                fields = oldline.split(':')
                if (fields[0] == WRDS_POSTGRES_HOST and
                        int(fields[1]) == WRDS_POSTGRES_PORT and
                        fields[2] == WRDS_POSTGRES_DB and
                        fields[3] == self.wrds_username):
                    return True
        else:
            return False
