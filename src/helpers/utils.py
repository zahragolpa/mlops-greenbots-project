import os
import shutil
import json
from deepchecks.tabular import Dataset


def create_temporary_dir_if_not_exists(tmp_dir_path:os.PathLike='tmp'):
    """creation of a temporary folder

    Args:
        tmp_dir_path (os.PathLike, optional): Path of the folder. Defaults to 'tmp'.
    
    """
    if not os.path.exists(tmp_dir_path):
        os.makedirs(tmp_dir_path)
    return tmp_dir_path


def clean_temporary_dir(tmp_dir_path:os.PathLike='tmp'):
    """delete the temporary folder

    Args:
        tmp_dir_path (os.PathLike, optional): Path of the folder. Defaults to 'tmp'.
    """
    if os.path.exists(tmp_dir_path):
        shutil.rmtree(tmp_dir_path)


def cameltosnake(camel_string: str) -> str:
    # If the input string is empty, return an empty string
    if not camel_string:
        return ""
    # If the first character of the input string is uppercase,
    # add an underscore before it and make it lowercase
    elif camel_string[0].isupper():
        return f"_{camel_string[0].lower()}{cameltosnake(camel_string[1:])}"
    # If the first character of the input string is lowercase,
    # simply return it and call the function recursively on the remaining string
    else:
        return f"{camel_string[0]}{cameltosnake(camel_string[1:])}"


def camel_to_snake(s: str):
    if len(s)<=1:
        return s.lower()
    # Changing the first character of the input string to lowercase
    # and calling the recursive function on the modified string
    return cameltosnake(s[0].lower()+s[1:])


def load_json(fpath):
    # JSON file
    with open(fpath, "r") as f:
        # Reading from file
        data = json.loads(f.read())
    return data


def get_categorical_cols(dataset_name):
    if dataset_name == "cardio":
        return ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_group', 'bmi', 'map']
    return NotImplementedError


def get_label_col(dataset_name):
    if dataset_name == "cardio":
        return "cardio"
    return NotImplementedError


def create_dc_dataset(dataset) -> Dataset:
    """
    Creates deepchecks datasets for given pandas dataframes wrt the categorical columns and the dataset label.
    Args:
        dataset: pd.DataFrame input dataset

    Returns:
        deepchecks_dataset: deepchecks Dataset
    """
    categorical_cols = get_categorical_cols(dataset_name="cardio")
    label_col = get_label_col(dataset_name="cardio")
    try:
        return Dataset(dataset, label=label_col, cat_features=categorical_cols)
    except:
        try:
            return Dataset(dataset, label=None, cat_features=categorical_cols)
        except Exception as e:
            print(f'Error creating deepchecks dataset: {e}')
