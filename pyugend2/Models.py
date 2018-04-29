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
        self.name = 'replication m'
        self.label = 'replication m'
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

    def load_optimistic_data_mgmt(self):
        '''
        This function will load the parameter values for the optimistic
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
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.300
        self.bf2 = 0.4
        self.bf3 = 0.300
        self.df1 = 0.056
        self.df2 = 0.00
        self.df3 = 0.146
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.112
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_most_optimistic_data_mgmt(self):
        '''
        This function will load the parameter values for the most optimistic
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
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.400
        self.bf2 = 0.400
        self.bf3 = 0.300
        self.df1 = 0.036
        self.df2 = 0.00
        self.df3 = 0.054
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.112
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_pessimistic_data_mgmt(self):
        '''
        This function will load the parameter values for the pessimistic
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
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.172
        self.bf2 = 0.4
        self.bf3 = 0.167
        self.df1 = 0.106
        self.df2 = 0.050
        self.df3 = 0.124
        self.dm1 = 0.069
        self.dm2 = 0.057
        self.dm3 = 0.076
        self.phire2 = 0.125
        self.phire3 = 0.150
        self.female_promotion_probability_1 = 0.0555
        self.female_promotion_probability_2 = 0.1905
        self.male_promotion_probability_1 = 0.0635
        self.male_promotion_probability_2 = 0.1149
        self.upperbound = 84
        self.lowerbound = 64
        self.variation_range = 3

    def load_baseline_data_science(self):
        '''
        This function will sets the parameter values for the model
        to the baseline for the science department.
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.310
        self.bf2 = 0.222
        self.bf3 = 0.0
        self.df1 = 0.0
        self.df2 = 0.0
        self.df3 = 0.017
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.033
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def load_optimistic_data_science(self):
        '''
        This function will load the parameter values for the optimistic
        scenario of the Science Department into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void
        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.310
        self.bf2 = 0.322
        self.bf3 = 0.050
        self.df1 = 0.0
        self.df2 = 0.0
        self.df3 = 0.069
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.085
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def load_most_optimistic_data_science(self):
        '''
        This function will load the parameter values for the most optimistic
        scenario of the Science Department into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.400
        self.bf2 = 0.322
        self.bf3 = 0.100
        self.df1 = 0.0
        self.df2 = 0.0
        self.df3 = 0.0
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.085
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

    def load_pessimistic_data_science(self):
        '''
        This function will load the parameter values for the pessimistic
        scenario of the Science Department into the model
        :return: This function does not return anything. It changes the
        current model in place.
        :rtype: void

        '''

        self.nf1 = 14
        self.nf2 = 13
        self.nf3 = 9
        self.nm1 = 37
        self.nm2 = 28
        self.nm3 = 239
        self.vac3 = 8.31
        self.vac2 = 5.9
        self.vac1 = 5.303
        self.bf1 = 0.310
        self.bf2 = 0.222
        self.bf3 = 0.0
        self.df1 = 0.050
        self.df2 = 0.050
        self.df3 = 0.043
        self.dm1 = 0.009
        self.dm2 = 0.017
        self.dm3 = 0.059
        self.phire2 = 0.158
        self.phire3 = 0.339
        self.female_promotion_probability_1 = 0.122
        self.female_promotion_probability_2 = 0.188
        self.male_promotion_probability_1 = 0.19
        self.male_promotion_probability_2 = 0.19
        self.upperbound = 350
        self.lowerbound = 330
        self.variation_range = 3

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

    def run_parameter_sweep(self, number_of_runs, param, llim,
                            ulim, num_of_steps):

        '''
        This function sweeps a single parameter and captures the effect of
        that variation on the overall model. Any valid parameter can be chosen.

        :param number_of_runs: The number of model iterations per parameter
        value
        :type number_of_runs: int
        :param param: The name of the parameter to sweep
        :type param: basestring
        :param llim: lower limit of the parameter value
        :type llim: float
        :param ulim: upper limit of the parameter value
        :type ulim: float
        :param num_of_steps: total number of increments in the range between
        the upper and lower limits.
        :type num_of_steps: int
        :return: a Dataframe containing individual model runs using the
        parameter increments
        :rtype: pandas Dataframe
        '''

        # First I will create a structured array to hold the results of the
        # simulation. The size of the array should be one for each step in the
        # parameter sweep. To calculate that,

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)

        parameter_sweep_results = pd.DataFrame(np.zeros([len(
            parameter_sweep_increments),
            len(RESULTS_COLUMNS + FEMALE_MATRIX_COLUMNS[1:]) + 1]))

        parameter_sweep_results.columns = ['increment'] + RESULTS_COLUMNS + \
                                          FEMALE_MATRIX_COLUMNS[1:]
        parameter_sweep_results.loc[:, 'increment'] = parameter_sweep_increments

        # Run simulations with parameter increments and collect into a container.

        for i, val in enumerate(parameter_sweep_increments):
            setattr(self, param, val)
            self.run_multiple(number_of_runs)

            # Sets the values the sweep data matrix to the last values in the
            #  multiple runs results_matrix.
            parameter_sweep_results.iloc[i, 1: neg(len(
                FEMALE_MATRIX_COLUMNS)) - 1] = self.results_matrix.tail(1).iloc[
                                               0, 1: -1]

            # Sets the year in the sweep data matrix to the last year in the
            # results_matrix.
            parameter_sweep_results.iloc[i, 1] = self.results_matrix.tail(
                1).iloc[0, len(RESULTS_COLUMNS)]

            # Fills the sweep matrix with data from the female percentage
            # matrices
            parameter_sweep_results.iloc[i, len(RESULTS_COLUMNS) + 1:] = \
                self.pct_female_matrix.tail(1).iloc[0, 1:]

        self.parameter_sweep_results = parameter_sweep_results

        # BEGIN BLOCK
        # Reset the models to original settings. This is very important,
        # otherwise settings from the parameter sweep will contaminate
        # subsequent runs of the model.

        self.load_baseline_data_mgmt()
        # END BLOCK

        return (0)

    def run_probability_parameter_sweep_overall(self,
                                                number_of_runs,
                                                param,
                                                llim,
                                                ulim,
                                                num_of_steps,
                                                target):

        '''

        This function sweeps a single parameter and captures the effect of
        that variation on the probability of achieving a target at the end of the model.
        Any valid parameter can be chosen.

        :param number_of_runs: The number of model iterations per parameter
        value
        :type number_of_runs: int
        :param param: The name of the parameter to sweep
        :type param: basestring
        :param llim: lower limit of the parameter value
        :type llim: float
        :param ulim: upper limit of the parameter value
        :type ulim: float
        :param num_of_steps: total number of increments in the range between
        the upper and lower limits.
        :type num_of_steps: int
        :return: a Dataframe containing individual model runs using the
        parameter increments
        :rtype: pandas Dataframe
        '''

        # First I will create a structured array to hold the results of the
        # simulation. The size of the array should be one for each step in the
        # parameter sweep. To calculate that,

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)
        parameter_sweep_columns = ['increment',
                                   'Year',
                                   'Probability',
                                   'Mean',
                                   'Min',
                                   'Max']
        parameter_sweep_results = pd.DataFrame(np.zeros([len(
            parameter_sweep_increments),
            len(parameter_sweep_columns)]))

        parameter_sweep_results.columns = parameter_sweep_columns
        parameter_sweep_results.loc[:, 'increment'] = parameter_sweep_increments

        # Run simulations with parameter increments and collect into a container.

        for i, val in enumerate(parameter_sweep_increments):
            setattr(self, param, val)
            self.run_probability_analysis_gender_proportion(number_of_runs,
                                                            target)
            parameter_sweep_results.iloc[i, 1:] = self.probability_matrix.tail(
                1).iloc[0, 1:]

        self.probability_sweep_results = parameter_sweep_results

        # BEGIN BLOCK
        # Reset the models to original settings. This is very important,
        # otherwise settings from the parameter sweep will contaminate
        # subsequent runs of the model.

        self.load_baseline_data_mgmt()
        # END BLOCK

        return (0)

    def run_probability_analysis_gender_proportion(self, num_runs, target):

        ## First run the model multiple times to generate the mean and standard deviation matrices. This will also create the res_array attribute for the stored simulation data.

        self.run_multiple(num_runs)

        ## Then I have to create an array to hold the probability for the target value given the array at that time. So I would take in the target value and a vector of current values.

        probability_matrix = pd.DataFrame(np.zeros([self.duration, 5]))
        probability_matrix.columns = ['Year', 'Probability', 'Mean', 'Min',
                                      'Max']
        probability_matrix['Year'] = list(range(self.duration))

        ## Pull the gender ratio data from the sims and extract probability of reaching the target.

        for idx in range(self.duration):
            _s = np.array([sum(list([r['run']['number_f1'][idx],
                                     r['run']['number_f2'][idx],
                                     r['run']['number_f3'][idx]])) / sum(
                list([r['run']['number_f1'][idx],
                      r['run']['number_f2'][idx],
                      r['run']['number_f3'][idx],
                      r['run']['number_m1'][idx],
                      r['run']['number_m2'][idx],
                      r['run']['number_m3'][idx]])) for r in self.res_array])
            probability_matrix.loc[idx, 'Probability'] = \
                calculate_empirical_probability_of_value(target, _s)
            probability_matrix.loc[idx, 'Mean'] = _s.mean()
            probability_matrix.loc[idx, 'Min'] = _s.min()
            probability_matrix.loc[idx, 'Max'] = _s.max()

        self.probability_matrix = probability_matrix

    def run_probability_analysis_gender_by_level(self, num_runs, target):

        ## First run the model multiple times to generate the mean and standard deviation matrices. This will also create the res_array attribute for the stored simulation data.

        self.run_multiple(num_runs)

        probability_by_level_data = pd.DataFrame(np.zeros([self.duration, 7]))
        probability_by_level_data.columns = ['year', 'pf1', 'pf2', 'pf3', 'pm1',
                                             'pm2', 'pm3']

        for idx in range(self.duration):
            _u1 = np.array([r['run']['number_f1'][idx] / (r['run']['number_f1'][
                                                              idx] + r[
                                                              'run'][
                                                              'number_m1'][idx])
                            for r in self.res_array])

            _u2 = np.array([r['run']['number_f2'][idx] / (r['run']['number_f2'][
                                                              idx] + r[
                                                              'run'][
                                                              'number_m2'][idx])
                            for r in self.res_array])

            _u3 = np.array([r['run']['number_f3'][idx] / (r['run']['number_f3'][
                                                              idx] + r[
                                                              'run'][
                                                              'number_m3'][idx])
                            for r in self.res_array])

            probability_by_level_data['year'] = idx

            probability_by_level_data.loc[idx, 'pf1'] = \
                calculate_empirical_probability_of_value(target, _u1)

            probability_by_level_data.loc[idx, 'pf2'] = \
                calculate_empirical_probability_of_value(target, _u2)

            probability_by_level_data.loc[idx, 'pf3'] = \
                calculate_empirical_probability_of_value(target, _u3)

            probability_by_level_data.loc[idx, 'pm1'] = \
                1 - probability_by_level_data['pf1'][idx]

            probability_by_level_data.loc[idx, 'pm2'] = \
                1 - probability_by_level_data['pf2'][idx]

            probability_by_level_data.loc[idx, 'pm3'] = \
                1 - probability_by_level_data['pf3'][idx]

        self.probability_by_level = probability_by_level_data

        return (probability_by_level_data)

    def run_probability_analysis_parameter_sweep_gender_proportion(self,
                                                                   number_of_runs,
                                                                   param, llim,
                                                                   ulim,
                                                                   num_of_steps,
                                                                   target):

        pass

    def run_probability_analysis_parameter_sweep_gender_detail(self,
                                                               number_of_runs,
                                                               param,
                                                               prof_group, llim,
                                                               ulim,
                                                               num_of_steps,
                                                               target):

        ## This is similar to the parameter sweep, except that I am capturing the probability instead of
        # the mean and standard deviation

        ## Setup the sweep increments

        parameter_sweep_increments = np.linspace(llim, ulim, num_of_steps)

        ## Now I create a container for the data. In this case I am only looking at the probability a certain count
        ## is equal to or greater than a particular target value.

        empirical_probability_param_sweep_df = pd.DataFrame(
            np.zeros([len(parameter_sweep_increments),
                      len(PROBABILITY_ARRAY_COLUMN_NAMES)]),
            columns=PROBABILITY_ARRAY_COLUMN_NAMES)

        ## Loop over all increments and get the results for the final year. Then pass these results to the probability
        ## calculation function to get the empirical probability that the value is the target or greater.

        for i, val in enumerate(parameter_sweep_increments):
            setattr(self, param, val)
            self.run_multiple(number_of_runs)
            model_final_year_results = self.pd_last_row_data

            empirical_probability_param_sweep_df['param'][i] = val

            empirical_probability_param_sweep_df['prof_group_mean'][i] = \
                model_final_year_results[prof_group].mean()

            empirical_probability_param_sweep_df['probability'][
                i] = calculate_empirical_probability_of_value(target,
                                                              model_final_year_results[
                                                                  prof_group])

        self.last_empirical_probability_detail = empirical_probability_param_sweep_df

        return (empirical_probability_param_sweep_df)

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
