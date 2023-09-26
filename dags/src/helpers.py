import uuid
import os
import json
import pickle
import pandas as pd
import logging

import config

def generate_uuid()->str:
    """
    Generate a random UUID
    :return: str
    """
    
    return str(uuid.uuid4()).replace('-','')


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Save a dataset.

    Args:
        df: The DataFrame to be saved.
        path: The path where the dataset should be saved.

    Returns:
        None
    """
    df.to_csv(path, index=False)
    logging.info("Dataset saved to: %s", path)
    print(f"[INFO] Dataset saved to {path}")

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset.

    Args:
        path: The path of the dataset to be loaded.

    Returns:
        The loaded DataFrame.
    """
    logging.info("Dataset loaded %s", path)
    return pd.read_csv(path)


def save_model_as_pickle(model, model_name, directory=None) -> None:
    """
    Save a model as a pickle file.

    Args:
        model: The model object to be saved.
        model_name: The name of the model.
        directory: The directory where the pickle file should be saved. 
                   If not provided, the default models directory will be used.

    Returns:
        None
    """
    if directory:
        filename = os.path.join(directory, model_name + ".pkl")
    else:
        filename = os.path.join(config.PATH_DIR_MODELS, model_name + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    
    print(f"[INFO] Model saved as pickle file: {filename}")
    logging.info("Model saved as pickle file: %s", filename)
    
    
def load_model_from_pickle(model_name: str):
    """
    Load a pickle model.

    Args:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model object.
    """
    with open(os.path.join(config.PATH_DIR_MODELS, model_name + ".pkl"), "rb") as f:
        logging.info("Model loaded: %s", model_name)
        print(f"[INFO] Model loaded: {model_name}")
        return pickle.load(f)



def save_model_as_json(model:dict, model_name:str, directory:str=None):
    """
    Save a model as a json file.

    Args:
        model: The model object to be saved.
        model_name: The name of the model.
        directory: The directory where the pickle file should be saved. 
                   If not provided, the default models directory will be used.

    Returns:
        None
    """
    
    if directory:
        filename = os.path.join(directory, model_name+".json")
    else:
        filename = os.path.join(config.PATH_DIR_MODELS, model_name+".json")
    with open(filename, "w") as f:
        json.dump(model, f)
        
    logging.info("Model saved as json file: %s", filename)
    print("[INFO] Model saved as json file:", filename)

def load_model_from_json(model_name: str) -> dict:
    """
    Load a json model.

    Args:
        model_name: The name of the model to be loaded.

    Returns:
        dict.
    """
    with open(os.path.join(config.PATH_DIR_MODELS, model_name+".json"), "r") as f:
        logging.info("Model loaded: %s", model_name)
        print(f"[INFO] Model loaded: {model_name}")
        return json.load(f)
    

def check_dataset_sanity(df: pd.DataFrame) -> bool:
    """
    Checks the sanity of a dataset by identifying null values in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be checked for null values.

    Returns:
    bool: True if the dataset is considered sane (no null values), False otherwise.
    
    Raises:
    Exception: If there are null values in the DataFrame, an exception is raised with the column names containing null values.
    """
    # Check for null values in the DataFrame
    nulls = df.isnull().sum()
    
    # Get the column names with null values
    null_columns = nulls[nulls > 0].index.tolist()
    
    # If there are no null values, return True
    if len(null_columns) == 0:
        return True
    else:
        # If there are null values, raise an exception with the column names
        logging.exception("There are null values in the training dataset: %s", null_columns)
        raise Exception(f"There are null values in the training dataset: {null_columns}")



def check_dataset_sanity(df: pd.DataFrame) -> bool:
    """
    Checks the sanity of a dataset by identifying null values in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be checked for null values.

    Returns:
    bool: True if the dataset is considered sane (no null values), False otherwise.
    
    Raises:
    Exception: If there are null values in the DataFrame, an exception is raised with the column names containing null values.
    """
    # Check for null values in the DataFrame
    nulls = df.isnull().sum()
    
    # Get the column names with null values
    null_columns = nulls[nulls > 0].index.tolist()
    
    # If there are no null values, return True
    if len(null_columns) == 0:
        return True
    else:
        # If there are null values, raise an exception with the column names
        logging.exception("There are null values in the training dataset: %s", null_columns)
        raise Exception(f"There are null values in the training dataset: {null_columns}")


def persist_deploy_report(job_id: str, model_name: str) -> None:
    """
    Persist the deploy report of a job.

    Parameters:
    job_id (str): The ID of the job.
    model_name (str): The name of the model.

    Returns:
    None
    """
    # Create a dictionary representing the deploy report
    report = {
        "job_id": job_id,
        "purpose_to_int": f"{job_id}_purpose_to_int_model.json",
        "missing_values": f"{job_id}_missing_values_model.pkl",
        "prediction_model": f"{model_name}.pkl",
        "train_report": f"{job_id}_train_report.json",
    }
    
    # Save the deploy report as a JSON file
    json.dump(report, open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "w"))
    
    # Print the path where the deployment report is saved
    logging.info("Deployment report saved as : %s", os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"))
    print(f'[INFO] Deployment report saved as {os.path.join(config.PATH_DIR_MODELS, "deploy_report.json")}')







