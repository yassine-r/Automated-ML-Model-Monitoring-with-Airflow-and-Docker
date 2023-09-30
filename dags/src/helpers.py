import uuid
import os
import json
import pickle
import pandas as pd
import datetime
import glob
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import config
import queries

credentials = json.load(open(config.PATH_TO_CREDENTIALS, 'r'))
engine = create_engine(f"postgresql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}")


def generate_uuid()->str:
    """
    Generate a random UUID
    :return: str
    """
    
    return str(uuid.uuid4()).replace('-','')

def get_model_type(job_id:str) -> str:
    """
    Get the type of a model.
    :param job_id: str
    :return: str
    """
    report_filename = os.path.join(config.PATH_DIR_MODELS, f"{job_id}_train_report.json")
    return json.load(open(report_filename, "r"))["final_model"]


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
    print(f"[INFO] Dataset saved to {path}")

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset.

    Args:
        path: The path of the dataset to be loaded.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_csv(path)

def locate_raw_data_filename(job_id:str) -> str:
    """
    Locate the raw data file.
    :param job_id: str
    :return: str
    """

    files = glob(os.path.join(config.PATH_DIR_DATA, "collected", f"{job_id}_*.csv"))
    if len(files) == 0:
        print(f"[WARNING] No raw data file found for job_id : {job_id}.")
        return None
    return files[0]

def locate_preprocessed_filenames(job_id:str) -> dict:
    """
    Locate the preprocessed data files.
    :param job_id: str
    :return: dict
    """
    files = sorted(glob(os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_*.csv")))
    if len(files) == 0:
        raise(Exception(f"No preprocessed data file found for job_id : {job_id}."))
    elif len(files) > 2:
        raise(Exception(f"More than one preprocessed data file found for job_id : {job_id} ->\n{files}"))
    elif len(files) == 1:
        training_filename = None
        inference_filename = list(filter(lambda x: "inference" in x, files))[0]
        return training_filename, inference_filename
    else:
        training_filename = list(filter(lambda x: "training" in x, files))[0]
        inference_filename = list(filter(lambda x: "inference" in x, files))[0]
        return training_filename, inference_filename

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
    
    
def load_model_from_pickle(model_name: str):
    """
    Load a pickle model.

    Args:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model object.
    """
    with open(os.path.join(config.PATH_DIR_MODELS, model_name + ".pkl"), "rb") as f:
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
    print(f'[INFO] Deployment report saved as {os.path.join(config.PATH_DIR_MODELS, "deploy_report.json")}')



def create_table_ml_job():
    """
    Create a table in the database.
    :return: None
    """
    engine.execute(text(queries.CREATE_TABLE_ML_JOB).execution_options(autocommit=True))
    print(f"[INFO] Table {credentials['database']}.mljob ready!")

def create_table_mlreport():
    raise(NotImplementedError)

def get_latest_deployed_job_id(status:str="pass") -> str:
    """
    Get the latest deployed job id by looking for the latest of all jobs with stage `deploy` and the specified status.
    :param status: str
    :return: str
    """
    try:
        return json.load(open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"))).get("job_id")
    except Exception as e:
        assert status in config.STATUS, f"[ERROR] Status `{status}` is not valid! Choose from {config.STATUS}"
        query = text(queries.GET_LATEST_DEPLOYED_JOB_ID.format(status=status))
        r = pd.read_sql(query, engine)
        if r.shape[0] == 0:
            return None
        return str(r['job_id'].values[0])
    
def log_activity(job_id:str, job_type:str, stage:str, status:str, message:str, job_date:datetime.date=None):
    """
    Logs the activity of a job.
    :param job_id: str
    :param job_type: str
    :param stage: str
    :param status: str
    :param message: str
    :param job_date: datetime.date
    :return: None
    """
    assert stage in config.STAGES, f"[ERROR] Stage `{stage}` is not valid! Choose from {config.STAGES}"
    assert status in config.STATUS, f"[ERROR] Status `{status}` is not valid! Choose from {config.STATUS}"
    assert job_type in config.JOB_TYPES, f"[ERROR] Job type `{job_type}` is not valid! Choose from {config.JOB_TYPES}"
    message = message.replace("'", "\\")
    engine.execute(text(queries.LOG_ACTIVITY.format(job_id=str(job_id), job_type=job_type, stage=str(stage), status=str(status), message=message, job_date=job_date)).execution_options(autocommit=True))
    print(f"[INFO] Job {job_id} logged as {job_type}::{stage}::{status}::{message}")