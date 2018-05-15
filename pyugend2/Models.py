"""
Base Model Module
-----------------

This is the base class for all model modules. This class does not contain an particular model but it does include all of the functions to run a model, capture model statistics, and visualize model data.

"""

__author__ = 'krishnab'
__version__ = '0.1.0'

import abc
import numpy as np
import pandas as pd
import datetime
from operator import neg
from .DataManagement import DataManagement


## Initialize Constants

PROFESSOR_LEVEL_NAMES = list(['f1n', 'f2n', 'f3n', 'm1n', 'm2n', 'm3n'])

PROBABILITY_ARRAY_COLUMN_NAMES = list(
    ['param', 'prof_group_mean', 'probability'])

LEVELS = list(['number_f1',
               'number_f2',
               'number_f3',
               'number_m1',
               'number_m2',
               'number_m3'])





np.seterr(divide='ignore', invalid='ignore')


class Base_model(metaclass=abc.ABCMeta):
    def __init__(self, argsdict):
        self.name = 'base model m'
        self.label = 'base model m'
        self.nf1 = argsdict.get('number_of_females_1', 0)
        self.nf2 = argsdict.get('number_of_females_2', 0)
        self.nf3 = argsdict.get('number_of_females_3', 0)
        self.nm1 = argsdict.get('number_of_males_1', 0)
        self.nm2 = argsdict.get('number_of_males_2', 0)
        self.nm3 = argsdict.get('number_of_males_3',0)
        self.bf1 = argsdict.get('hiring_rate_women_1', 0)
        self.bf2 = argsdict.get('hiring_rate_women_2',0)
        self.bf3 = argsdict.get('hiring_rate_women_3',0)
        self.df1 = argsdict.get('attrition_rate_women_1',0)
        self.df2 = argsdict.get('attrition_rate_women_2',0)
        self.df3 = argsdict.get('attrition_rate_women_3',0)
        self.dm1 = argsdict.get('attrition_rate_men_1',0)
        self.dm2 = argsdict.get('attrition_rate_men_2',0)
        self.dm3 = argsdict.get('attrition_rate_men_3',0)
        self.duration = argsdict.get('duration',0)
        self.female_promotion_probability_1 = argsdict.get('female_promotion_probability_1',0)
        self.female_promotion_probability_2 = argsdict.get('female_promotion_probability_2',0)
        self.male_promotion_probability_1 = argsdict.get('male_promotion_probability_1',0)
        self.male_promotion_probability_2 = argsdict.get('male_promotion_probability_2',0)
        self.upperbound = argsdict.get('upperbound',0)
        self.lowerbound = argsdict.get('lowerbound',0)
        self.variation_range = argsdict.get('variation_range',0)
        self.mgmt_data = DataManagement()
        self.model_run_date_time = get_date_time_of_today()
        self.model_common_name = argsdict.get('model_name', '')
        self.number_of_sim_columns = 0
        self.itercount = 0

    def load_baseline_data_mgmt(self):
        '''
        This function will load the parameter values for the baseline
        scenario of the Business School into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void
        '''

        self.nf1 = 3
        self.nf2 = 3
        self.nf3 = 2
        self.nm1 = 11
        self.nm2 = 12
        self.nm3 = 43
        self.vac3 = 0
        self.vac2 = 0
        self.vac1 = 0
        self.bf1 = 0.172
        self.bf2 = 0.4
        self.bf3 = 0.167
        self.df1 = 0.056
        self.df2 = 0.00
        self.df3 = 0.074
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.040
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.female_promotion_probability_1 = 0.0556
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0611
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3
        self.name = "Promote-Hire baseline"
        self.label = "Promote-Hire baseline"

    @abc.abstractmethod
    def run_model(self):
        pass

    @abc.abstractmethod
    def get_number_of_model_data_columns(self):
        pass


    def run_multiple(self,num_iterations):

        # first get the sizing of the storage array
        # set up the array to hold the data array from each model run.
        self.run_model_iterations(num_iterations)
        # calculate summaries
        summary_matrix = self.calculate_simulation_summaries()

        return simulation_matrix, summary_matrix
    def run_model_iterations(self, num_iterations):

        if self.number_of_sim_columns == 0:
            raise ValueError('number of simulation columns should not be 0. '
                             'Fix the __init__ function in the model class.')
        results_matrix = np.zeros((num_iterations, self.duration, self.number_of_sim_columns))
        self.itercount = 0
        for i in num_iterations:
            results_matrix.loc[i, :, :] = self.run_model()
            self.itercount += 1
        self.simulation_matrix = results_matrix
        self.itercount = 0

    def calculate_simulation_summaries(self):

        # allocate column names
        all_columns = set(self.simulation_matrix.columns)
        summary_matrix_columns,\
        sim_results_cols, \
        sim_setting_cols, \
        mgmt_data_cols  = self.create_summary_column_names_list()

        # create new dataframe to hold summary info
        temp = np.zeros([self.duration, summary_column_list])
        summary_matrix = pd.DataFrame(temp)
        summary_matrix.columns = summary_column_list

        for c in sim_results_cols:
            for i in self.duration:
                summary_matrix.loc[i, c + '_avg'] = self.simulation_matrix.loc[:, i, c].mean()
                summary_matrix.loc[i, c + '_std'] = self.simulation_matrix.loc[:, i, c].std()
                summary_matrix.loc[i, c + '_975'] = np.percentile(self.simulation_matrix.loc[:, i, c], 97.5)
                summary_matrix.loc[i, c + '_025'] = np.percentile(self.simulation_matrix.loc[:, i, c], 2.5)

        for c in sim_setting_cols:
            for i in self.duration:
                summary_matrix.loc[i, c] = self.simulation_matrix.loc[:, i, c].mean()

        for c in mgmt_data_cols:
            summary_matrix.loc[0:self.mgmt_data.shape[0], c] = self.mgmt_data.loc[:, c]

        self.summary_matrix = summary_matrix



    def create_summary_column_names_list(self, all_columns):

        columns_to_summarize = {x for x in simulation_matrix.columns if 'ss_' not in x}
        simulation_settings_columns = all_columns - columns_to_summarize
        mgmt_data_columns = set(self.mgmt_data.columns)
        column_list = []
        # add simulation results columns to list
        for c in columns_to_summarize:
            column_list.append([c + '_avg',
                               c + '_std',
                               c + '_975',
                               c + '_025'])

        # add simulation settings and management column names to list
        column_list.append(simulation_settings_columns)
        column_list.append(mgmt_data_columns)
        return column_list, columns_to_summarize, simulation_settings_columns, mgmt_data_columns



## Supplementary/Helper functions

def calculate_empirical_probability_of_value(criterion, data_vector):
    emp_prob = 1 - sum(data_vector <= criterion) / len(data_vector)
    return (emp_prob)

def get_date_time_of_today():
    return datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
