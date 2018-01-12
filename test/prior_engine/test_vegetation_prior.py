import vegetation_prior

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"

# path_to_json_file = './test/test_data/test_data_stores.yml'


def test_get_mean_state_vector():
    vegetation_prior.get_mean_state_vector()


    # data_access_component = DataAccessComponent()
    # data_stores = data_access_component.read_data_stores(path_to_json_file)
    # assert_equal(2, len(data_stores))
