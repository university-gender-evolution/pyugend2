"""
Base Model Module
-----------------

This is the base class for all model modules. This class does not contain an particular model but it does include all of the functions to run a model, capture model statistics, and visualize model data.

"""

__author__ = 'krishnab'
__version__ = '0.1.0'

import numpy as np
import pandas as pd
import datetime
from operator import neg
from .ColumnSpecs import MODEL_RUN_COLUMNS, EXPORT_COLUMNS_FOR_CSV
from .ColumnSpecs import RESULTS_COLUMNS, FEMALE_MATRIX_COLUMNS
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


class Base_model():
    def __init__(self, argsdict):
        self.name = 'base model m'
        self.label = 'base model m'
        self.nf1 = argsdict['number_of_females_1']
        self.nf2 = argsdict['number_of_females_2']
        self.nf3 = argsdict['number_of_females_3']
        self.nm1 = argsdict['number_of_males_1']
        self.nm2 = argsdict['number_of_males_2']
        self.nm3 = argsdict['number_of_males_3']
        self.vac3 = 0
        self.vac2 = 0
        self.vac1 = 0
        self.bf1 = argsdict['hiring_rate_women_1']
        self.bf2 = argsdict['hiring_rate_women_2']
        self.bf3 = argsdict['hiring_rate_women_3']
        self.df1 = argsdict['attrition_rate_women_1']
        self.df2 = argsdict['attrition_rate_women_2']
        self.df3 = argsdict['attrition_rate_women_3']
        self.dm1 = argsdict['attrition_rate_men_1']
        self.dm2 = argsdict['attrition_rate_men_2']
        self.dm3 = argsdict['attrition_rate_men_3']
        self.phire1 = 1
        self.phire2 = 1
        self.phire3 = 1
        self.duration = argsdict['duration']
        self.female_promotion_probability_1 = argsdict['female_promotion_probability_1']
        self.female_promotion_probability_2 = argsdict['female_promotion_probability_2']
        self.male_promotion_probability_1 = argsdict['male_promotion_probability_1']
        self.male_promotion_probability_2 = argsdict['male_promotion_probability_2']
        self.upperbound = argsdict['upperbound']
        self.lowerbound = argsdict['lowerbound']
        self.variation_range = argsdict['variation_range']
        self.run = 0
        self.runarray = 0
        self.pd_last_row_data = 0
        self.pct_female_matrix = 0
        self.probability_matrix = 0
        self.probability_by_level = 0
        self.mgmt_data = DataManagement()

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


    def run_model(self):

        self.res = np.zeros([self.duration, 26], dtype=np.float32)
        df_ = pd.DataFrame(self.res)
        df_.columns = MODEL_RUN_COLUMNS

        recarray_results = df_.to_records(index=True)
        self.res = recarray_results
        return recarray_results

    def run_multiple(self, number_of_runs):

        res_array = np.recarray((number_of_runs,), dtype=[('run', object)])

        ## Then I need to run a loop to run multiple models and return their values to the record array

        for idx in range(number_of_runs):
            res_array['run'][idx] = self.run_model()

        ##Create empty arrays to hold the results

        self.results_matrix = pd.DataFrame(np.zeros([self.duration,
                                                     len(RESULTS_COLUMNS)]))
        self.results_matrix.columns = RESULTS_COLUMNS

        self.pct_female_matrix = pd.DataFrame(np.zeros([self.duration,
                                                        len(
                                                            FEMALE_MATRIX_COLUMNS)]))
        self.pct_female_matrix.columns = FEMALE_MATRIX_COLUMNS

        ## Collect mean and standard deviations for each column/row across all
        # iterations of the model.


        for idx in range(self.duration):

            # Set the year in the results matrix

            self.results_matrix.loc[idx, 0] = idx

            ## This section will iterate over all of the values in the results
            ## matrix for a year, and it will get the mean and average values
            ## for each statistic for that year. This info contains the raw
            ## numbers for each grouping and goes to the gender numbers plots.

            for k, f in enumerate(MODEL_RUN_COLUMNS):
                _t = np.array([r['run'][f][idx] for r in res_array])

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[k + 1]] = np.array(
                    np.mean(_t)) if \
                    np.isfinite(np.array(np.mean(_t))) else 0

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[k + 27]] = np.array(
                    np.std(_t)) if \
                    np.isfinite(np.array(np.std(_t))) else 0

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[
                                            k + 53]] = np.percentile(_t, 2.5) if \
                    np.isfinite(np.percentile(_t, 2.5))  else 0

                self.results_matrix.loc[idx,
                                        RESULTS_COLUMNS[
                                            k + 79]] = np.percentile(_t,
                                                                     97.5) if \
                    np.isfinite(np.percentile(_t, 97.5)) else 0

            # Calculate the mean and standard deviation/percentiles
            # for each grouping.

            for l, lev in enumerate(LEVELS):
                if l <= 2:

                    _num = np.array([r['run'][LEVELS[l]][idx] for r in
                                     res_array])
                    _denom = np.array([r['run'][LEVELS[l]][idx] + r['run'][
                        LEVELS[l + 3]][idx] for r in res_array])
                    _u = np.nan_to_num(np.divide(_num, _denom))


                else:

                    _num = np.array([r['run'][LEVELS[l]][idx] for r in
                                     res_array])
                    _denom = np.array([r['run'][LEVELS[l]][idx] + r['run'][
                        LEVELS[l - 3]][idx] for r in res_array])
                    _u = np.nan_to_num(np.divide(_num, _denom))

                self.pct_female_matrix.loc[idx, 'year'] = idx

                self.pct_female_matrix.loc[
                    idx, FEMALE_MATRIX_COLUMNS[2 * l + 1]] \
                    = np.nanmean(_u)

                self.pct_female_matrix.loc[
                    idx, FEMALE_MATRIX_COLUMNS[2 * l + 2]] \
                    = np.nanstd(_u)

                self.pct_female_matrix.loc[idx,
                                           FEMALE_MATRIX_COLUMNS[
                                               12 + 2 * l + 1]] = np.nanpercentile(
                    _u,
                    2.5) if \
                    np.isfinite(np.nanpercentile(_u, 2.5)) else 0

                self.pct_female_matrix.loc[idx,
                                           FEMALE_MATRIX_COLUMNS[
                                               12 + 2 * l + 2]] = np.nanpercentile(
                    _u,
                    97.5) if \
                    np.isfinite(np.nanpercentile(_u, 97.5)) else 0

        self.res_array = res_array

    def export_model_run(self, model_label, model_choice, number_of_runs):

        if not hasattr(self, 'res'):
            self.run_multiple(number_of_runs)

        # first I will allocate the memory by creating an empty dataframe.
        # then I will iterate over the res_array matrix and write to the
        # correct rows of the dataframe. This is more memory efficient compared
        # to appending to a dataframe.

        # print(pd.DataFrame(self.res_array['run'][3]))

        columnnames = ['run', 'year'] + MODEL_RUN_COLUMNS + \
                      EXPORT_COLUMNS_FOR_CSV + ['model_name']

        print_array = np.zeros([self.duration * number_of_runs,
                                len(columnnames)])

        for idx in range(number_of_runs):
            print_array[(idx * self.duration):(idx * self.duration +
                                               self.duration), 0] = idx

            print_array[(idx * self.duration):(idx * self.duration +
                                               self.duration),
            1:-1] = pd.DataFrame(self.res_array['run'][idx])

        # work with barbara to craft the filename
        # model_label + 160114_HH:MM(24hour) +

        filename = model_label + "_" + str(datetime.datetime.now()) + "_iter" \
                   + str(number_of_runs) + ".csv"

        df_print_array = pd.DataFrame(print_array, columns=columnnames).round(2)
        df_print_array.iloc[:, -1] = model_choice
        df_print_array.to_csv(filename)


## Supplementary/Helper functions

def calculate_empirical_probability_of_value(criterion, data_vector):
    emp_prob = 1 - sum(data_vector <= criterion) / len(data_vector)
    return (emp_prob)
