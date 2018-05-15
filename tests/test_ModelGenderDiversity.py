import pytest
from pyugend2.ModelGenderDiversity import Model3GenderDiversity
from pyugend2.Comparison import Comparison

@pytest.mark.usefixtures('mgmt_data')
@pytest.mark.usefixtures('mock_data')

def test_model_run(mgmt_data):
    model = Model3GenderDiversity(mgmt_data)
    model.init_default_hiring_rate()
    res = model.run_model()
    res.to_csv('ModelGenderDiversity_test_export.csv', header=True, index=False)
    print(res)

def test_model_multiple_run(mgmt_data):

    model = Model3GenderDiversity(mgmt_data)
    model.init_default_hiring_rate()
    model.run_multiple(100)
    model.summary_matrix.to_csv('model_summary.csv', index=False, header=True)
