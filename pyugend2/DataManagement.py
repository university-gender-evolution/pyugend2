
from .abcDepartmentData import abcDepartmentData
import pandas as pd
from . import dirFinder
import os
data_dir = os.path.dirname(os.path.abspath(dirFinder.__file__))

class DataManagement(abcDepartmentData):

    def __init__(self):
        super().__init__()
        self.data = self.load_data()

    def load_data(self):

        return pd.read_csv(data_dir+'/mgmt_data.csv')


    def get_data(self):
        return self.data

    def get_field(self, field):
        return self.data.loc[:, field].tolist()

