"""
Stochastic Model Gender Diversity 5 Year Forecasts
---------------------

This is the second generation of the gender diversity model
with 5 year forecasts.




Notes:
09/14/2017 - This is the first version of the second generate model.
"""
__author__ = 'krishnab'
__version__ = '0.1.0'
import numpy as np
import pandas as pd
from numpy.random import binomial, multinomial
from .Models import Base_model
from .ModelGenderDiversity import Model3GenderDiversity
import itertools
import math
from .ColumnSpecs import MODEL_RUN_COLUMNS, EXPORT_COLUMNS_FOR_CSV



class ModelGenderDiversityGrowthForecastIncremental(Model3GenderDiversity):
    def __init__(self, argsdict):
        Base_model.__init__(self, argsdict)
        self.name = "model-3-5-year-growth-forecast"
        self.label = "model-3-5-year-growth-forecast"

    def init_growth_rate(self, candidate):
        self.growth_forecasts = candidate

    def calculate_yearly_dept_size_targets(self):

        repeat_interval = self.__calculate_forecast_interval()
        result = list(itertools.chain.from_iterable(itertools.repeat(x, repeat_interval) for x in self.growth_forecasts))
        return result

    def __calculate_forecast_interval(self):

        return math.ceil(self.duration/len(self.growth_forecasts))

    def __calculate_dept_size_forecast_vector(self, initial_dept_size, forecast_interval):

        dept_size_vector = [initial_dept_size]

        for k,v in enumerate(forecast_interval):
            dept_size_vector.append(math.ceil(dept_size_vector[k]*(1+forecast_interval[k])))
        return dept_size_vector

    def __calculate_upperbound_vector(self, dept_size_vector):
        dept_upper_bound = [self.upperbound]
        department_change = [y-x for x,y in zip(dept_size_vector, dept_size_vector[1:])]
        for k,v in enumerate(department_change):
            dept_upper_bound.append(dept_upper_bound[k] + department_change[k])
        return dept_upper_bound

    def __calculate_lowerbound_vector(self, dept_size_vector):
        dept_lower_bound = [self.lowerbound]
        department_change = [y-x for x,y in zip(dept_size_vector, dept_size_vector[1:])]
        for k,v in enumerate(department_change):
            dept_lower_bound.append(dept_lower_bound[k] + department_change[k])
        return dept_lower_bound

    def run_model(self):

        # initialize data structure

        self.res = np.zeros([self.duration,
                             len(MODEL_RUN_COLUMNS) +
                             len(EXPORT_COLUMNS_FOR_CSV)],
                            dtype=np.float32)
        self.res[0, 0] = self.nf1
        self.res[0, 1] = self.nf2
        self.res[0, 2] = self.nf3
        self.res[0, 3] = self.nm1
        self.res[0, 4] = self.nm2
        self.res[0, 5] = self.nm3
        self.res[0, 6] = self.vac3
        self.res[0, 7] = self.vac2
        self.res[0, 8] = self.vac1
        self.res[0, 9] = self.female_promotion_probability_1
        self.res[0, 10] = self.female_promotion_probability_2
        self.res[0, 11] = self.res[0, 0:3].sum()/self.res[0, 0:6].sum()
        self.res[0, 12] = 0
        self.res[0, 13] = self.res[0, 0:6].sum()
        self.res[0, 14:] = 0
        initial_department_size = self.res[0, 0:6].sum()

        # I assign the state variables to temporary variables. That way I
        # don't have to worry about overwriting the original state variables.

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
        unfilled_vacanies = 0
        extra_vacancies=0
        department_size_forecasts = self.calculate_yearly_dept_size_targets()
        department_size_target = self.__calculate_dept_size_forecast_vector(initial_department_size,
                                                                department_size_forecasts)
        dept_upperbound = self.__calculate_upperbound_vector(department_size_target)
        dept_lowerbound = self.__calculate_lowerbound_vector(department_size_target)

        for i in range(1, self.duration):
            # initialize variables for this iteration

            prev_number_of_females_level_1 = self.res[i - 1, 0]
            prev_number_of_females_level_2 = self.res[i - 1, 1]
            prev_number_of_females_level_3 = self.res[i - 1, 2]
            prev_number_of_males_level_1 = self.res[i - 1, 3]
            prev_number_of_males_level_2 = self.res[i - 1, 4]
            prev_number_of_males_level_3 = self.res[i - 1, 5]
            department_size = self.res[i - 1, 0:6].sum()
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
            self.res[i, 0] = self.res[i-1, 0] - female_attrition_level_1
            self.res[i, 1] = self.res[i-1, 1] - female_attrition_level_2
            self.res[i, 2] = self.res[i-1, 2] - female_attrition_level_3
            self.res[i, 3] = self.res[i-1, 3] - male_attrition_level_1
            self.res[i, 4] = self.res[i-1, 4] - male_attrition_level_2
            self.res[i, 5] = self.res[i-1, 5] - male_attrition_level_3

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
            promotions_of_females_level_2_3 = binomial(self.res[i, 1],
                                                       female_promotion_probability_2_3)
            promotions_of_males_level_2_3 = binomial(self.res[i, 4],
                                                     male_promotion_probability_2_3)
            promotions_of_females_level_1_2 = binomial(self.res[i, 0],
                                                       female_promotion_probability_1_2)
            promotions_of_males_level_1_2 = binomial(self.res[i, 3],
                                                     male_promotion_probability_1_2)

            # update model numbers
            # add promotions to levels
            self.res[i, 1] += promotions_of_females_level_1_2
            self.res[i, 2] += promotions_of_females_level_2_3
            self.res[i, 4] += promotions_of_males_level_1_2
            self.res[i, 5] += promotions_of_males_level_2_3

            # remove the promoted folks from previous level
            self.res[i, 0] -= promotions_of_females_level_1_2
            self.res[i, 1] -= promotions_of_females_level_2_3
            self.res[i, 3] -= promotions_of_males_level_1_2
            self.res[i, 4] -= promotions_of_males_level_2_3

            # hiring of new faculty
            hires = multinomial(total_vacancies,
                                [self.hiring_rate_f1,
                                 self.hiring_rate_f2,
                                 self.hiring_rate_f3,
                                 self.hiring_rate_m1,
                                 self.hiring_rate_m2,
                                 self.hiring_rate_m3])

            self.res[i, 0] += hires[0]
            self.res[i, 1] += hires[1]
            self.res[i, 2] += hires[2]
            self.res[i, 3] += hires[3]
            self.res[i, 4] += hires[4]
            self.res[i, 5] += hires[5]

            # fill in summary info for model run

            self.res[i, 6] = sum(list([
                male_attrition_level_3,
                female_attrition_level_3]))

            self.res[i, 7] = sum(list([
                male_attrition_level_2,
                female_attrition_level_2]))

            self.res[i, 8] = sum(list([
                male_attrition_level_1,
                female_attrition_level_1]))

            self.res[i, 9] = 0
            self.res[i, 10] = 0
            self.res[i, 11] = self.res[i, 0:3].sum()/self.res[i,0:6].sum()
            unfilled_vacanies = abs(department_size - self.res[i, 0:6].sum())
            self.res[i, 12] = unfilled_vacanies
            department_size = self.res[i, 0:6].sum()
            self.res[i, 13] = department_size
            self.res[i, 14] = hires[2]
            self.res[i, 15] = hires[5]
            self.res[i, 16] = hires[1]
            self.res[i, 17] = hires[4]
            self.res[i, 18] = hires[0]
            self.res[i, 19] = hires[3]
            self.res[i, 20] = 0
            self.res[i, 21] = 0
            self.res[i, 22] = promotions_of_females_level_2_3
            self.res[i, 23] = promotions_of_males_level_2_3
            self.res[i, 24] = promotions_of_females_level_1_2
            self.res[i, 25] = promotions_of_males_level_1_2
            self.res[i, 26] = self.hiring_rate_f1
            self.res[i, 27] = self.hiring_rate_f2
            self.res[i, 28] = self.hiring_rate_f3
            self.res[i, 29] = self.hiring_rate_m1
            self.res[i, 30] = self.hiring_rate_m2
            self.res[i, 31] = self.hiring_rate_m3
            self.res[i, 32] = attrition_rate_female_level_1
            self.res[i, 33] = attrition_rate_female_level_2
            self.res[i, 34] = attrition_rate_female_level_3
            self.res[i, 35] = attrition_rate_male_level_1
            self.res[i, 36] = attrition_rate_male_level_2
            self.res[i, 37] = attrition_rate_male_level_3
            self.res[i, 38] = 0
            self.res[i, 39] = 0
            self.res[i, 40] = 0
            self.res[i, 41] = female_promotion_probability_1_2
            self.res[i, 42] = female_promotion_probability_2_3
            self.res[i, 43] = male_promotion_probability_1_2
            self.res[i, 44] = male_promotion_probability_2_3
            self.res[i, 45] = department_size_upper_bound
            self.res[i, 46] = department_size_lower_bound
            self.res[i, 47] = variation_range
            self.res[i, 48] = self.duration


            department_growth = department_size_target[i] - department_size
            department_size_upper_bound = dept_upperbound[i] + department_growth
            department_size_lower_bound = dept_lowerbound[i] + department_growth
            # matching wise [(-1, 1), (-1, 1), (0, 2)]
            new_department_size = department_size + department_growth

            flag = False
            while flag == False:
                changes = np.random.choice([-1, 0, 1], variation_range)

                if (new_department_size + changes.sum() <=
                    department_size_upper_bound and new_department_size +
                    changes.sum() >= department_size_lower_bound):
                    extra_vacancies = department_growth + changes.sum()
                    flag = True

                if (new_department_size > department_size_upper_bound):
                    extra_vacancies = -1*variation_range
                    flag = True

                if department_size < department_size_lower_bound:
                    extra_vacancies = department_growth+variation_range
                    flag = True

        df_ = pd.DataFrame(self.res)
        df_.columns = MODEL_RUN_COLUMNS + EXPORT_COLUMNS_FOR_CSV
        recarray_results = df_.to_records(index=True)
        self.run = recarray_results
        return recarray_results

