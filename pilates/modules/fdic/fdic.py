"""
Provides processing functions for the FDIC data (directly from FDIC, not WRDS)
"""

from pilates import data_module


class fdic(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
