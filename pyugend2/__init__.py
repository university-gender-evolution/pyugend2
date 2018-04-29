# -*- coding: utf-8 -*-

"""Top-level package for pyugend2."""

__author__ = """krishnab bhogaonker"""
__email__ = 'cyclotomiq@gmail.com'
__version__ = '0.1.0'
__all__ = ['Base_model',
           'Model3GenderDiversity',
           'Comparison']



from .Models import Base_model
from .ModelGenderDiversity import Model3GenderDiversity
from .Comparison import Comparison
from .ModelGenderDiversityLinearGrowth import ModelGenderDiversityLinearGrowth
from .ModelGenderDiversityGrowthForecast import ModelGenderDiversityGrowthForecast
from .ModelGenderDiversityGrowthForecastIncrementalChange import ModelGenderDiversityGrowthForecastIncremental
from .DataManagement import DataManagement
