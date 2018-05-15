"""
Stochastic Model Gender Diversity
---------------------

This is the second generation of the gender diversity model.





Notes:
09/14/2017 - This is the first version of the second generate model.
"""
__author__ = 'krishnab'
__version__ = '0.1.0'
import numpy as np
import pandas as pd
from numpy.random import binomial, multinomial
from .Models import Base_model
from .ColumnSpecs import MODEL_RUN_COLUMNS




class Model3GenderDiversity(Base_model):
    def __init__(self, argsdict):
        Base_model.__init__(self, argsdict)
        self.model_common_name = argsdict.get('model_name', 'model_3_baseline_no_growth')
        self.hiring_rate_f1 = argsdict.get('hiring_rate_f1',0)
        self.hiring_rate_f2 = argsdict.get('hiring_rate_f2',0)
        self.hiring_rate_f3 = argsdict.get('hiring_rate_f3',0)
        self.hiring_rate_m1 = argsdict.get('hiring_rate_m1',0)
        self.hiring_rate_m2 = argsdict.get('hiring_rate_m2',0)
        self.hiring_rate_m3 = argsdict.get('hiring_rate_m3',0)
        self.number_of_sim_columns, self.sim_column_list = self.get_number_of_model_data_columns()

    def init_default_hiring_rate(self):

        self.hiring_rate_f1 = 5/40
        self.hiring_rate_f2 = 2/40
        self.hiring_rate_f3 = 1/40
        self.hiring_rate_m1 = 24/40
        self.hiring_rate_m2 = 3/40
        self.hiring_rate_m3 = 5/40


    def get_number_of_model_data_columns(self):
        run = self.run_model()
        return len(run.columns), run.columns

    def run_model(self):

        # initialize data structure

        tres = np.zeros([self.duration,
                        len(MODEL_RUN_COLUMNS)],
                        dtype=np.float32)
        res = pd.DataFrame(tres)
        res.columns = MODEL_RUN_COLUMNS

        male_columns = ['m1', 'm2', 'm3']
        female_columns = ['f1', 'f2','f3']
        all_faculty = male_columns + female_columns

        ########################################################
        ## Write all simulation parameter values and initial values
        ## to the dataframe that captures model state and settings.
        ## I will export this dataframe at the end of the simulation.
        ########################################################

        res.loc[0, 'f1'] = self.nf1
        res.loc[0, 'f2'] = self.nf2
        res.loc[0, 'f3'] = self.nf3
        res.loc[0, 'm1'] = self.nm1
        res.loc[0, 'm2'] = self.nm2
        res.loc[0, 'm3'] = self.nm3
        res.loc[0, 'lev1'] = self.nf1 + self.nm1
        res.loc[0, 'lev2'] = self.nf2 + self.nm2
        res.loc[0, 'lev3'] = self.nf3 + self.nm3
        res.loc[0, 'f'] = self.nf1 + self.nf2 + self.nf3
        res.loc[0, 'm'] = self.nm1 + self.nm2 + self.nm3
        res.loc[0, 'fpct1'] = self.nf1/res.loc[0, 'lev1']
        res.loc[0, 'fpct2'] = self.nf2/res.loc[0, 'lev2']
        res.loc[0, 'fpct3'] = self.nf3/res.loc[0, 'lev3']
        res.loc[0, 'mpct1'] = 1 - res.loc[0, 'fpct1']
        res.loc[0, 'mpct2'] = 1 - res.loc[0, 'fpct2']
        res.loc[0, 'mpct3'] = 1 - res.loc[0, 'fpct3']
        res.loc[0, 'fpct'] = res.loc[0, female_columns].sum()/res.loc[0, all_faculty].sum()
        res.loc[0, 'deptn'] = res.loc[0, all_faculty].sum()
        res.loc[0, 'ss_fhire1_r'] = self.hiring_rate_f1
        res.loc[0, 'ss_fhire2_r'] = self.hiring_rate_f2
        res.loc[0, 'ss_fhire3_r'] = self.hiring_rate_f3
        res.loc[0, 'ss_mhire1_r'] = self.hiring_rate_m1
        res.loc[0, 'ss_mhire2_r'] = self.hiring_rate_m2
        res.loc[0, 'ss_mhire3_r'] = self.hiring_rate_m3
        res.loc[0, 'ss_fattr1_r'] = self.df1
        res.loc[0, 'ss_fattr2_r'] = self.df2
        res.loc[0, 'ss_fattr3_r'] = self.df3
        res.loc[0, 'ss_mattr1_r'] = self.dm1
        res.loc[0, 'ss_mattr2_r'] = self.dm2
        res.loc[0, 'ss_mattr3_r'] = self.dm3
        res.loc[0, 'ss_fprom1_r'] = self.female_promotion_probability_1
        res.loc[0, 'ss_fprom2_r'] = self.female_promotion_probability_2
        res.loc[0, 'ss_mprom1_r'] = self.male_promotion_probability_1
        res.loc[0, 'ss_mprom2_r'] = self.male_promotion_probability_2
        res.loc[0, 'ss_deptn_ub'] = self.upperbound
        res.loc[0, 'ss_deptn_lb'] = self.lowerbound
        res.loc[0, 'ss_deptn_range'] = self.variation_range
        res.loc[0, 'ss_duration'] = self.duration
        res.loc[0, 'ss_date'] = self.model_run_date_time
        res.loc[0, 'ss_model'] = self.model_common_name
        res.loc[0, 'ss_run'] = self.itercount
        res.loc[0, 'a_ss_yr'] = 0
        res.loc[0, 'hire'] = 0
        res.loc[0, 'unfild'] = 0
        #############################################################

        # I assign the state variables to temporary variables. That way I
        # don't have to worry about overwriting the original state variables.
        # The more descriptive variable names also make for more understandable
        # and readable code.

        attrition_rate_female_level_1 = self.df1
        attrition_rate_female_level_2 = self.df2
        attrition_rate_female_level_3 = self.df3
        attrition_rate_male_level_1 = self.dm1
        attrition_rate_male_level_2 = self.dm2
        attrition_rate_male_level_3 = self.dm3
        female_promotion_probability_1_2 = self.female_promotion_probability_1
        female_promotion_probability_2_3 = self.female_promotion_probability_2
        male_promotion_probability_1_2 = self.male_promotion_probability_1
        male_promotion_probability_2_3 = self.male_promotion_probability_2
        department_size_upper_bound = self.upperbound
        department_size_lower_bound = self.lowerbound
        variation_range = self.variation_range
        extra_vacancies=0

        ###################################################################
        ## Execute main simulation loop
        ##
        ###################################################################


        for i in range(1, self.duration):
            # initialize variables for this iteration

            prev_number_of_females_level_1 = res.loc[i - 1, 'f1']
            prev_number_of_females_level_2 = res.loc[i - 1, 'f2']
            prev_number_of_females_level_3 = res.loc[i - 1, 'f3']
            prev_number_of_males_level_1 = res.loc[i - 1, 'm1']
            prev_number_of_males_level_2 = res.loc[i - 1, 'm2']
            prev_number_of_males_level_3 = res.loc[i - 1, 'm3']
            department_size = res.loc[i - 1, all_faculty].sum()

            # Process Model
            # attrition process
            female_attrition_level_1 = binomial(prev_number_of_females_level_1,
                                                attrition_rate_female_level_1)
            female_attrition_level_2 = binomial(prev_number_of_females_level_2,
                                                attrition_rate_female_level_2)
            female_attrition_level_3 = binomial(prev_number_of_females_level_3,
                                                attrition_rate_female_level_3)
            male_attrition_level_1 = binomial(prev_number_of_males_level_1,
                                              attrition_rate_male_level_1)
            male_attrition_level_2 = binomial(prev_number_of_males_level_2,
                                              attrition_rate_male_level_2)
            male_attrition_level_3 = binomial(prev_number_of_males_level_3,
                                              attrition_rate_male_level_3)
            # update model numbers
            res.loc[i, 'f1'] = res.loc[i-1, 'f1'] - female_attrition_level_1
            res.loc[i, 'f2'] = res.loc[i-1, 'f2'] - female_attrition_level_2
            res.loc[i, 'f3'] = res.loc[i-1, 'f3'] - female_attrition_level_3
            res.loc[i, 'm1'] = res.loc[i-1, 'm1'] - male_attrition_level_1
            res.loc[i, 'm2'] = res.loc[i-1, 'm2'] - male_attrition_level_2
            res.loc[i, 'm3'] = res.loc[i-1, 'm3'] - male_attrition_level_3

            # get total number of vacancies based on attrition
            subtotal_vacancies_1 = female_attrition_level_1 \
                + male_attrition_level_1
            subtotal_vacancies_2 = female_attrition_level_2 \
                + male_attrition_level_2
            subtotal_vacancies_3 = female_attrition_level_3 \
                + male_attrition_level_3
            total_vacancies = subtotal_vacancies_3 \
                + subtotal_vacancies_2 + subtotal_vacancies_1

            total_vacancies = max(total_vacancies+extra_vacancies, 0)

            # process promotions
            promotions_of_females_level_2_3 = binomial(res.loc[i, 'f2'],
                                        female_promotion_probability_2_3)
            promotions_of_males_level_2_3 = binomial(res.loc[i, 'm2'],
                                        male_promotion_probability_2_3)
            promotions_of_females_level_1_2 = binomial(res.loc[i, 'f1'],
                                        female_promotion_probability_1_2)
            promotions_of_males_level_1_2 = binomial(res.loc[i, 'm1'],
                                        male_promotion_probability_1_2)

            # update model numbers
            # add promotions to levels
            res.loc[i, 'f2'] += promotions_of_females_level_1_2
            res.loc[i, 'f3'] += promotions_of_females_level_2_3
            res.loc[i, 'm2'] += promotions_of_males_level_1_2
            res.loc[i, 'm3'] += promotions_of_males_level_2_3

            # remove the promoted folks from previous level
            res.loc[i, 'f1'] -= promotions_of_females_level_1_2
            res.loc[i, 'f2'] -= promotions_of_females_level_2_3
            res.loc[i, 'm1'] -= promotions_of_males_level_1_2
            res.loc[i, 'm2'] -= promotions_of_males_level_2_3

            # hiring of new faculty
            hires = multinomial(total_vacancies,
                              [self.hiring_rate_f1,
                               self.hiring_rate_f2,
                               self.hiring_rate_f3,
                               self.hiring_rate_m1,
                               self.hiring_rate_m2,
                               self.hiring_rate_m3])

            res.loc[i, 'f1'] += hires[0]
            res.loc[i, 'f2'] += hires[1]
            res.loc[i, 'f3'] += hires[2]
            res.loc[i, 'm1'] += hires[3]
            res.loc[i, 'm2'] += hires[4]
            res.loc[i, 'm3'] += hires[5]

            # fill in summary info for model run

            # capture attrition level 3
            res.loc[i, 'attr1'] = male_attrition_level_1 + female_attrition_level_1
            res.loc[i, 'attr2'] = male_attrition_level_2 + female_attrition_level_2
            res.loc[i, 'attr3'] = male_attrition_level_3 + female_attrition_level_3
            res.loc[i, 'fattr1'] = female_attrition_level_1
            res.loc[i, 'fattr2'] = female_attrition_level_2
            res.loc[i, 'fattr3'] = female_attrition_level_3
            res.loc[i, 'mattr1'] = male_attrition_level_1
            res.loc[i, 'mattr2'] = male_attrition_level_2
            res.loc[i, 'mattr3'] = male_attrition_level_3
            res.loc[i, 'fattr'] = sum([female_attrition_level_1,
                                       female_attrition_level_2,
                                       female_attrition_level_3])
            res.loc[i, 'mattr'] = sum([male_attrition_level_1,
                                       male_attrition_level_2,
                                       male_attrition_level_3])

            # capture gender proportion for department
            res.loc[i, 'fpct'] = round(res.loc[i, female_columns].sum()/res.loc[i,all_faculty].sum(), 3)
            res.loc[i, 'fpct1'] = round(res.loc[i, 'f1']/res.loc[i, ['f1', 'm1']].sum(), 3)
            res.loc[i, 'fpct2'] = round(res.loc[i, 'f2']/res.loc[i, ['f2', 'm2']].sum(), 3)
            res.loc[i, 'fpct3'] = round(res.loc[i, 'f3']/res.loc[i, ['f3', 'm3']].sum(), 3)
            res.loc[i, 'mpct1'] = round(1 - res.loc[i, 'fpct1'], 3)
            res.loc[i, 'mpct2'] = round(1 - res.loc[i, 'fpct2'], 3)
            res.loc[i, 'mpct3'] = round(1 - res.loc[i, 'fpct3'], 3)

            # capture number of unfilled vacancies as the department size in
            # the last time-step versus the current department size (summing
            # all professor groups). If there is a difference then some
            # vacancies were not filled. This is not a good metric to monitor
            # when using a growth model because the department size is supposed
            # to change from time-step to timestep.

            res.loc[i, 'unfild'] = abs(department_size - res.loc[i, all_faculty].sum())

            # capture the current department size.
            department_size = res.loc[i, all_faculty].sum()
            res.loc[i, 'deptn'] = res.loc[i, all_faculty].sum()
            res.loc[i, 'lev1'] = res.loc[i, ['f1', 'm1']].sum()
            res.loc[i, 'lev2'] = res.loc[i, ['f2', 'm2']].sum()
            res.loc[i, 'lev3'] = res.loc[i, ['f3', 'm3']].sum()
            res.loc[i, 'f'] = res.loc[i, female_columns].sum()
            res.loc[i, 'm'] = res.loc[i, male_columns].sum()
            # capture the number of hires for each group.
            res.loc[i, 'fhire1'] = hires[0]
            res.loc[i, 'fhire2'] = hires[1]
            res.loc[i, 'fhire3'] = hires[2]
            res.loc[i, 'mhire1'] = hires[3]
            res.loc[i, 'mhire2'] = hires[4]
            res.loc[i, 'mhire3'] = hires[5]
            res.loc[i, 'hire1'] = res.loc[i, ['fhire1', 'mhire1']].sum()
            res.loc[i, 'hire2'] = res.loc[i, ['fhire2', 'mhire2']].sum()
            res.loc[i, 'hire3'] = res.loc[i, ['fhire3', 'mhire3']].sum()
            res.loc[i, 'fhire'] = sum([hires[0], hires[1], hires[2]])
            res.loc[i, 'mhire'] = sum([hires[3], hires[4], hires[5]])
            res.loc[i, 'hire'] = sum([hires[0], hires[1], hires[2],
                                       hires[3], hires[4], hires[5]])
            # capture promotions for each group. Since we cannot
            # have promotions from level 3 (full professor), these are set to
            # zero by default.
            res.loc[i, 'fprom1'] = promotions_of_females_level_1_2
            res.loc[i, 'mprom1'] = promotions_of_males_level_1_2
            res.loc[i, 'fprom2'] = promotions_of_females_level_2_3
            res.loc[i, 'mprom2'] = promotions_of_males_level_2_3
            res.loc[i, 'prom1'] = res.loc[i, ['fprom1', 'mprom1']].sum()
            res.loc[i, 'prom2'] = res.loc[i, ['fprom2', 'mprom2']].sum()
            res.loc[i, 'fprom'] = res.loc[i, ['fprom1', 'fprom2']].sum()
            res.loc[i, 'mprom'] = res.loc[i, ['mprom1', 'mprom2']].sum()
            res.loc[i, 'prom'] = res.loc[i, ['fprom', 'mprom']].sum()
            # capture the hiring rate parameters for each group
            res.loc[i, 'ss_fhire1_r'] = self.hiring_rate_f1
            res.loc[i, 'ss_fhire2_r'] = self.hiring_rate_f2
            res.loc[i, 'ss_fhire3_r'] = self.hiring_rate_f3
            res.loc[i, 'ss_mhire1_r'] = self.hiring_rate_m1
            res.loc[i, 'ss_mhire2_r'] = self.hiring_rate_m2
            res.loc[i, 'ss_mhire3_r'] = self.hiring_rate_m3

            # capture the attrition rate parameters for each group
            res.loc[i, 'ss_fattr1_r'] = attrition_rate_female_level_1
            res.loc[i, 'ss_fattr2_r'] = attrition_rate_female_level_2
            res.loc[i, 'ss_fattr3_r'] = attrition_rate_female_level_3
            res.loc[i, 'ss_mattr1_r'] = attrition_rate_male_level_1
            res.loc[i, 'ss_mattr2_r'] = attrition_rate_male_level_2
            res.loc[i, 'ss_mattr3_r'] = attrition_rate_male_level_3

            # capture the promotion probabilities for each group
            res.loc[i, 'ss_fprom1_r'] = female_promotion_probability_1_2
            res.loc[i, 'ss_fprom2_r'] = female_promotion_probability_2_3
            res.loc[i, 'ss_mprom1_r'] = male_promotion_probability_1_2
            res.loc[i, 'ss_mprom2_r'] = male_promotion_probability_2_3

            # capture the department size bounds and variation ranges.
            res.loc[i, 'ss_deptn_ub'] = department_size_upper_bound
            res.loc[i, 'ss_deptn_lb'] = department_size_lower_bound
            res.loc[i, 'ss_deptn_range'] = variation_range
            res.loc[i, 'ss_date'] = self.model_run_date_time
            res.loc[i, 'ss_model'] = self.model_common_name
            # capture the model duration, or the number of time-steps
            res.loc[i, 'ss_duration'] = self.duration
            res.loc[i, 'a_ss_yr'] = i
            res.loc[i, 'ss_run'] = self.itercount

            # Churn process:
            # Set the flag to False. When an allowable number of extra
            # FTEs are generated, then the flag will change to True,
            # and the loop will exit.
            flag = False
            while flag == False:

                # generate a vector of additional FTEs. The number of possible.
                # additions is based upon the ~variation_range~ variable.
                changes = np.random.choice([-1, 0, 1], variation_range)

                # Given the new vector, we need to test whether the current
                # department size plus additional changes will be within
                # the department upper and lower bound. If the new department
                # size exceeds the upperbound or falls under the lower bound,
                # then the random ~changes~ data vector is thrown out and
                # re-run. This process continues until a suitable value is found,
                # that keeps the department within the right size.

                # if the number of changes keeps the department within the proper
                # range, then add the extra changes to the total number of vacancies
                # for the next timestep of the model.
                if (department_size + changes.sum() <=
                        department_size_upper_bound and department_size +
                    changes.sum() >= department_size_lower_bound):
                    extra_vacancies = changes.sum()
                    flag = True

                # if the current department size is above the upper bound, then
                # set the number of extra FTEs to zero and exit the loop.
                if (department_size > department_size_upper_bound):
                    extra_vacancies = -1*variation_range
                    flag = True

                # if the current department size is below the lower bound, then
                # set the number of additional FTEs to the ~variation_range~ meaning
                # the maximum possible number of extra FTEs--in order to bring the
                # department size quickly back over the lower bound.
                if department_size < department_size_lower_bound:
                    extra_vacancies = variation_range
                    flag = True

        return res


