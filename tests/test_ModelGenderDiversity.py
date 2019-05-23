import pytest
from pyugend2.ModelGenderDiversity import Model3GenderDiversity
from pyugend2.Comparison import Comparison
from pyugend2.ModelGenderDiversityGrowth import ModelGenderDiversityGrowth
from pyugend2.ModelGenderDiversityGrowthAsDrift import ModelGenderDiversityGrowthAsDrift

@pytest.mark.usefixtures('mgmt_data')
@pytest.mark.usefixtures('sci_data')
@pytest.mark.usefixtures('mgmt_growth_data')

def test_model_run(mgmt_data):
    model = Model3GenderDiversity(mgmt_data)
    res = model.run_model()
    res.to_csv('ModelGenderDiversity_test_export.csv', header=True, index=False)

def test_model_multiple_run(mgmt_data):

    model = Model3GenderDiversity(mgmt_data)
    model.run_multiple(100)
    model.summary_matrix.to_csv('no_growth_model_summary.csv', index=False, header=True)

def test_growth_model_run(mgmt_growth_data):
    model = ModelGenderDiversityGrowth(mgmt_growth_data)
    res = model.run_model()
    res.to_csv('ModelGenderDiversityGrowth_test_export.csv', header=True, index=False)

def test_growth_model_multiple_run(mgmt_growth_data):

    model = ModelGenderDiversityGrowth(mgmt_growth_data)
    model.run_multiple(100)
    model.summary_matrix.to_csv('growth_model_summary.csv', index=False, header=True)

def test_growth_model_as_drift_multiple_run(mgmt_growth_data):
    model = ModelGenderDiversityGrowthAsDrift(mgmt_growth_data)
    model.run_multiple(10)
    model.summary_matrix.to_csv('growth_model_with_drift_summary.csv', index=False, header=True)

def test_annotation_to_model(mgmt_growth_data):
    model = ModelGenderDiversityGrowthAsDrift(mgmt_growth_data)
    model.run_multiple(10)
    model.summary_matrix.to_csv('growth_model_with_drift_summary.csv', index=False, header=True)
