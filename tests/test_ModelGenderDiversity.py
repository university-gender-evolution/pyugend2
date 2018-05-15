import pytest
from pyugend2.ModelGenderDiversity import Model3GenderDiversity

@pytest.mark.usefixtures('mgmt_data')
@pytest.mark.usefixtures('mock_data')

def test_model_run(mgmt_data):
    model = Model3GenderDiversity(mgmt_data)
    model.init_default_hiring_rate()
    res = model.run_model()
    res.to_csv('ModelGenderDiversity_test_export.csv', header=True)
    print(res)

