"""
Provides processing functions for the CFPB data.
"""

from pilates import data_module


class cfpb(data_module):

    def __init__(self, d):
        data_module.__init__(self, d)
