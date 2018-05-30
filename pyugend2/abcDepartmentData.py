

import abc

class abcDepartmentData(metaclass=abc.ABCMeta):

    def __init__(self):
        self.data = None

    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def get_field(self):
        pass
