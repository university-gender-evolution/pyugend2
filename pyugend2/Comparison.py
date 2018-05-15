__author__ = 'krishnab'
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from operator import add, sub
from .ColumnSpecs import MODEL_RUN_COLUMNS
import datetime


height = 700
width = 700
# CONSTANTS

# line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

class Comparison():
    def __init__(self, model_list):
        self.name = 'All Models'
        self.label = 'All Models'
        self.mlist = model_list

    def run_all_models(self, number_of_runs):
        for mod in self.mlist:
            mod.run_multiple(number_of_runs)


