from os.path import dirname, join

def get_test_data_path():
    """Returns the path to test datasets """
    return join(dirname(__file__), 'data')
