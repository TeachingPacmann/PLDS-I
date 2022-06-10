import yaml

def read_yaml(yaml_path):
    """
    Loader for yaml file.
    
    Args:
    - yaml_path(string): Path to yaml file.
    
    Returns:
    - params(dictionary): Dict ver of yaml file.
    """
    
    with open(yaml_path, "r") as stream:
        params = yaml.safe_load(stream)
    
    return params 