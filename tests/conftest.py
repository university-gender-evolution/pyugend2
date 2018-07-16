import pytest
from pyugend2.Comparison import Comparison
from pyugend2.ModelGenderDiversity import Model3GenderDiversity

@pytest.fixture(scope="module")
def sci_data():
    return ({'number_of_females_1': 14,
             'number_of_females_2': 3,
             'number_of_females_3': 19,
             'number_of_males_1': 37,
             'number_of_males_2': 28,
             'number_of_males_3': 239,
             'number_of_initial_vacancies_1': 0,
             'number_of_initial_vacancies_2': 0,
             'number_of_initial_vacancies_3': 0,
             'hiring_rate_women_1': 0.310,
             'hiring_rate_women_2': 0.222,
             'hiring_rate_women_3': 0,
             'attrition_rate_women_1': 0,
             'attrition_rate_women_2': 0,
             'attrition_rate_women_3': 0.017,
             'attrition_rate_men_1': 0.009,
             'attrition_rate_men_2': 0.017,
             'attrition_rate_men_3': 0.033,
             'probability_of_outside_hire_1': 1,
             'probability_of_outside_hire_2': 0.158,
             'probability_of_outside_hire_3': 0.339,
             'female_promotion_probability_1': 0.122,
             'female_promotion_probability_2': 0.188,
             'male_promotion_probability_1': 0.19,
             'male_promotion_probability_2': 0.19,
             'upperbound': 350,
             'lowerbound': 330,
             'variation_range': 3,
             'duration': 40})


@pytest.fixture(scope="module")
def mgmt_data():
    return ({'number_of_females_1': 3,
             'number_of_females_2': 3,
             'number_of_females_3': 2,
             'number_of_males_1': 11,
             'number_of_males_2': 12,
             'number_of_males_3': 43,
             'number_of_initial_vacancies_1': 0,
             'number_of_initial_vacancies_2': 0,
             'number_of_initial_vacancies_3': 0,
             'hiring_rate_women_1': 0.172,
             'hiring_rate_women_2': 0.4,
             'hiring_rate_women_3': 0.167,
             'attrition_rate_women_1': 0.056,
             'attrition_rate_women_2': 0.0001,
             'attrition_rate_women_3': 0.074,
             'attrition_rate_men_1': 0.069,
             'attrition_rate_men_2': 0.057,
             'attrition_rate_men_3': 0.040,
             'probability_of_outside_hire_1': 1,
             'probability_of_outside_hire_2': 0.125,
             'probability_of_outside_hire_3': 0.150,
             'female_promotion_probability_1': 0.0555,
             'female_promotion_probability_2': 0.1905,
             'male_promotion_probability_1': 0.0635,
             'male_promotion_probability_2': 0.1149,
             't_fpct': 0.15,
             'upperbound': 90,
             'lowerbound': 70,
             'variation_range': 3,
             'duration': 20})

@pytest.fixture(scope="module")
def mgmt_growth_data():
    return ({'number_of_females_1': 3,
             'number_of_females_2': 3,
             'number_of_females_3': 2,
             'number_of_males_1': 11,
             'number_of_males_2': 12,
             'number_of_males_3': 43,
             'number_of_initial_vacancies_1': 0,
             'number_of_initial_vacancies_2': 0,
             'number_of_initial_vacancies_3': 0,
             'hiring_rate_women_1': 0.172,
             'hiring_rate_women_2': 0.4,
             'hiring_rate_women_3': 0.167,
             'attrition_rate_women_1': 0.056,
             'attrition_rate_women_2': 0.0001,
             'attrition_rate_women_3': 0.074,
             'attrition_rate_men_1': 0.069,
             'attrition_rate_men_2': 0.057,
             'attrition_rate_men_3': 0.040,
             'probability_of_outside_hire_1': 1,
             'probability_of_outside_hire_2': 0.125,
             'probability_of_outside_hire_3': 0.150,
             'female_promotion_probability_1': 0.0555,
             'female_promotion_probability_2': 0.1905,
             'male_promotion_probability_1': 0.0635,
             'male_promotion_probability_2': 0.1149,
             't_fpct': 0.15,
             'upperbound': 90,
             'lowerbound': 70,
             'variation_range': 3,
             'duration': 20,
             'growth_rate': [.015, 0.015, 0, 0, 0]})

@pytest.fixture(scope="module")
def multi_model(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data),
                    ModelGenderDiversityLinearGrowth(**mgmt_data),
                    ModelGenderDiversityGrowthForecast(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    modlist[1].init_default_hiring_rate()
    modlist[1].init_growth_rate(0.01)
    modlist[2].init_default_hiring_rate()
    modlist[2].init_growth_rate([73, 78, 83, 88])
    return Comparison(modlist)

@pytest.fixture(scope="module")
def one_model(mgmt_data):
    modlist = list([Model3GenderDiversity(**mgmt_data)])
    modlist[0].init_default_hiring_rate()
    return Comparison(modlist)
