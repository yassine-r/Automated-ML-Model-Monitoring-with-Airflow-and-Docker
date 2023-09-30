import os
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
import helpers

cat_vars = list(map(str.lower, config.CAT_VARS))
num_vars = list(map(str.lower, config.NUM_VARS))
date_vars = list(map(str.lower, config.DATETIME_VARS))
exc_vars = list(map(str.lower, config.EXC_VARIABLES))

engineered_vars = {
    "categorical": ["application_year", "application_month", "application_week", "application_day", "application_season"],
    "numerical": ["current_credit_balance_ratio"],
    "date": ["application_date"]
}

def get_variables_with_missing_values(df: pd.DataFrame) -> list:
    """
    Get variables with missing values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        list: A list of variables with missing values.
    """
    # Calculate the count of missing values for each variable
    missing_counts = df.isnull().sum()
    
    # Filter the variables with missing values and return as a list
    variables_with_missing_values = missing_counts[missing_counts > 0].index.tolist()
    
    return variables_with_missing_values

def impute_missing_values(df: pd.DataFrame, method: str = "basic", mode: str = None, cat_vars: list = cat_vars, num_vars: list = num_vars, job_id: str = "") -> pd.DataFrame:
    """
    Treat missing values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The imputation method to use. Default is "basic".
        mode (str): The mode of operation, either "training" or "inference".
        cat_vars (list): List of categorical variables. Default is cat_vars.
        num_vars (list): List of numerical variables. Default is num_vars.
        job_id (str): The job ID. Default is an empty string.
        
    Returns:
        pd.DataFrame: The DataFrame with imputed missing values.
    """
    assert mode in ("training", "inference"), f"mode must be either 'training' or 'inference', but got {mode}"
    assert method in ["basic", "advanced"], f"{method} is not a valid method (basic, advanced)"
    
    if mode == "training":
        model = {
            "method": method,
            "imputes": dict()
        }
        
        for col in df.columns:
            print("[INFO] Treating missing values in column:", col)
            
            model["imputes"][col] = dict()
            
            if method == "basic":
                if col in set(cat_vars + engineered_vars["categorical"]):
                    model["imputes"][col]['mode'] = df[df[col].notnull()][col].mode()[0]
                elif col in set(num_vars + engineered_vars["numerical"]):
                    model["imputes"][col]['mean'] = df[df[col].notnull()][col].mean()
                elif col in set(date_vars + engineered_vars["date"]):
                    model["imputes"][col]['mode'] = df[df[col].notnull()][col].mode()[0]
                elif col in ["loan_id", "customer_id", "loan_status"] + exc_vars:
                    pass
                else:
                    raise ValueError(f"[ERROR] {col} is not a valid variable")
            
            if method == "advanced":
                raise NotImplementedError
        
        helpers.save_model_as_pickle(model, f"{job_id}_missing_values_model")
        return impute_missing_values(df, method=method, mode="inference", cat_vars=cat_vars, num_vars=num_vars, job_id=job_id)
    
    else:
        model = helpers.load_model_from_pickle(model_name=f"{job_id}_missing_values_model")
        cols = get_variables_with_missing_values(df)
        method = model["method"]
        
        if method == "basic":
            for col in cols:
                if col in set(cat_vars + engineered_vars["categorical"]):
                    df[col].fillna(model["imputes"][col]['mode'], inplace=True)
                elif col in set(num_vars + engineered_vars["numerical"]):
                    df[col].fillna(model["imputes"][col]['mean'], inplace=True)
                elif col in set(date_vars + engineered_vars["date"]):
                    df[col].fillna(model["imputes"][col]['mode'], inplace=True)
                elif col in ["loan_id", "customer_id", "loan_status"] + exc_vars:
                    pass
                else:
                    raise ValueError(f"[ERROR] {col} is not a valid variable. Pre-trained variables: {list(model['imputes'].keys())}")
        
        if method == "advanced":
            raise NotImplementedError
    
    return df

## enforcing datatypes

def enforce_numeric_to_float(x: str) -> float:
    """
    Converts a string to a float, or returns `np.nan` if the string is not a valid float.

    Parameters:
        x: The string to be converted to a float.

    Returns:
        The float representation of the string, or `np.nan` if the string could not be converted to a float.
    """
    
    try:
        return float(re.sub("[^0-9.]","", str(x)))
    except ValueError:
        return np.nan
    

def enforce_datatypes(df: pd.DataFrame, cat_vars: list=cat_vars, num_vars: list=num_vars) -> pd.DataFrame:
    """
    Enforces the data types of the columns in a pandas DataFrame.

    Args:
        df: The pandas DataFrame to be updated.
        cat_vars: A list of the categorical variables in the DataFrame.
        num_vars: A list of the numeric variables in the DataFrame.

    Returns:
        The pandas DataFrame with the enforced data types.
    """

    df["application_time"] = pd.to_datetime(df["application_time"])
    for var in num_vars:
        df[var] = df[var].apply(lambda x: enforce_numeric_to_float(x))
    for var in cat_vars:
        df[var] = df[var].astype(str)
    return df


## Encoding categorical features

def categorize_years_in_current_job(x: str) -> int:
    """
    Categorize years in current job into numerical values.
    
    Args:
    x (str): The input string representing years in current job.
    
    Returns:
    int: The categorized numerical value representing years in current job.
    """
    x = str(x).strip()  # Clean up input by removing leading/trailing spaces
    
    if x == '< 1 year':
        return 0  # Special case for less than 1 year
    
    if x in ('1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years'):
        return int(re.sub("[^0-9]", "", x))  # Extract numerical value from the string
    
    if x == '10+ years':
        return 11  # Special case for 10 or more years
    
    return -1  # Return -1 for unrecognized cases

def term_to_int(x: str) -> int:
    """
    Convert term (lowercased) to integer representation.
    
    Args:
    x (str): Lowercased term ("short term" or "long term").
    
    Returns:
    int: Integer representation of the term (0 for "short term", 1 for "long term"),
    or np.nan for unrecognized cases.
    """
    if x == "short term":
        return 0
    elif x == "long term":
        return 1
    else:
        return np.nan  # Return np.nan for unrecognized cases

def home_ownership_to_int(x: str) -> int:
    """
    Convert home ownership (lowercased) to integer representation.
    
    Args:
    x (str): Lowercased home ownership status.
    
    Returns:
    int: Integer representation of home ownership
    (0 for "rent", 1 for "mortgage", 2 for "own"),
    or np.nan for unrecognized cases.
    """
    if x == "rent":
        return 0
    elif "home mortgage" in x:
        return 1
    elif "own home" in x:
        return 2
    else:
        return np.nan  # Return np.nan for unrecognized cases
    

def train_purpose_to_int_model(x: pd.Series, method: str, job_id: str = "") -> dict:
    """
    Build a model file to be used to convert string variable `purpose` into integer datatype.
    
    Args:
        x (pd.Series): The input series.
        method (str): The method to use for conversion. Valid options are:
            - "ranking": Replace each value by its rank based on frequency.
            - "relative ranking": Replace each value by the ratio of its frequency to the highest frequency.
            - "weighted ranking": Replace each value by the ratio of its frequency to the total number of values (or sum of frequencies).
        job_id (str, optional): The job ID. Default is an empty string.
        
    Returns:
        dict: The purpose-to-int model.
    """
    assert method in ["ranking", "weighted ranking", "relative ranking"], f"{method} is not a valid methods (ranking, weighted ranking, relative ranking)"
    val_counts = x.value_counts()
    if method=="ranking":
        uniq_vals = sorted(val_counts.unique(), reverse=False)
        val_to_int = dict(zip(uniq_vals, range(1, len(uniq_vals)+1)))
        model = val_counts.apply(lambda x: val_to_int[x]).to_dict()
        helpers.save_model_as_json(model, f"{job_id}_purpose_to_int_model")        
        return model
    if method=="relative ranking":
        model = (val_counts/val_counts.max()).to_dict()
        helpers.save_model_as_json(model, f"{job_id}_purpose_to_int_model")        
        return model
    if method=="weighted ranking":
        model = (val_counts/val_counts.sum()).to_dict()
        helpers.save_model_as_json(model, f"{job_id}_purpose_to_int_model")        
        return model


def purpose_to_int(x:pd.Series, mode:str, method:str=None, model:str=None, job_id:str="") -> pd.Series:
    """
    Convert purpose to int.
    :param x:pd.Series
    :param mode: str, choose from "training", "inference"
    :param method: str, "ranking",  "weighted ranking", "relative ranking"
        - ranking 
            rank values by their frequency and assign a rank to each value. The most frequent value will have the highest rank
        - relative ranking
            replace each value by the ratio of its frequency to the highest frequency
        - weighted ranking
            replace each value by the ratio of its frequency to the total number of values
        when method is None and model is not None, any new value (not present in the model) will be encoded as 0
    :param model: method, model to predict the purpose. If None, a new model will be trained and saved to the default directory of models as defined in the config file
    :param save_report: bool, whether to save the report of missed/new values. Not implemented for nor
    :param job_id: str, job id
    :return:pd.Series
    """
    print("[INFO] Converting purpose to int using method:", method)
    
    if model==None:
        print("[INFO] No model for purpose-to-int conversion provided. Training a new model first...")
        mode = "training"
    if mode=="training":
        model = train_purpose_to_int_model(x, method, job_id=job_id)
        # return purpose_to_int(x, method=method, model=model, job_id=job_id)
        return x.apply(lambda x: model.get(x, 0))
    else:
        model = helpers.load_model_from_json(model_name=f"{job_id}_purpose_to_int_model")
        return x.apply(lambda x: model.get(x, 0))
    

def loan_status_to_int(x: str) -> int:
    """
    Convert loan status (lowercased) to integer representation.
    
    Args:
    x (str): Lowercased loan status.
    
    Returns:
    int: Integer representation of loan status
    (0 for "loan refused", 1 for "loan given"),
    or the input value itself if it's not recognized.
    """
    valid_statuses = ("loan given", "loan refused")
    assert x in valid_statuses or isinstance(x, int), f"{x} is not a valid loan status and is not an integer"
    
    x = str(x).strip()  # Clean up input by removing leading/trailing spaces
    
    if x == "loan refused":
        return 0
    if x == "loan given":
        return 1
    
    return x  # Return the original value if it's not recognized as a valid loan status


def encode_categorical_variables(df:pd.DataFrame, mode="training", purpose_encode_method="ranking", job_id:str="") -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame
        mode: str, "training" or "inference"
        purpose_encode_method: str, choose from "ranking", "weighted ranking", "relative ranking"
        job_id: str, job id
        
    Returns:
        DataFrame
    """
    assert mode in ("training", "inference"), f"{mode} is not a valid mode (training , inference)"
    assert isinstance(job_id, str)
    for col in cat_vars:
        assert col in df.columns, f"{col} not in {df.columns}"
        df[col] = df[col].str.lower()

    df["term"] = df["term"].apply(lambda x: term_to_int(x))
    df["home_ownership"] = df["home_ownership"].apply(lambda x: home_ownership_to_int(x))  
    df["years_in_current_job"] = df["years_in_current_job"].apply(lambda x: categorize_years_in_current_job(x))
    if config.TARGET.lower() in df.columns:
        df[config.TARGET.lower()] = df[config.TARGET.lower()].apply(lambda x: loan_status_to_int(x))
    df["purpose"] = purpose_to_int(df["purpose"], mode=mode, method=purpose_encode_method, job_id=job_id)
    return df

## features engineering

def month_to_season(month: int) -> int:
    """
    Convert month to season.
    
    Args:
    month (int): The input month (1-12).
    
    Returns:
    int: The corresponding season (1: Winter, 2: Spring, 3: Summer, 4: Fall).
    If the input month is not in the valid range (1-12), returns -1 to indicate an error.
    """
    if 1 <= month <= 3:
        return 1  # Winter
    elif 4 <= month <= 6:
        return 2  # Spring
    elif 7 <= month <= 9:
        return 3  # Summer
    elif 10 <= month <= 12:
        return 4  # Fall
    else:
        return np.nan  # Invalid month
    

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features based on the application_time column.
    
    Args:
    df (pd.DataFrame): The input DataFrame with application_time column.
    
    Returns:
    pd.DataFrame: The DataFrame with added engineered features.
    """
    # Check if the necessary columns are present in the DataFrame
    assert "application_time" in df.columns, f"application_time not in {df.columns}"
    
    # Extract date-based features from the application_time column
    df["application_date"] = df["application_time"].dt.date
    df["application_year"] = df["application_time"].dt.year
    df["application_month"] = df["application_time"].dt.month
    df["application_week"] = df["application_time"].dt.isocalendar().week  
    df["application_day"] = df["application_time"].dt.day
    
    # Map application_month to application_season using month_to_season function
    df["application_season"] = df["application_month"].apply(lambda x: month_to_season(x))
    
    # Calculate current_credit_balance_ratio while handling division by zero
    df["current_credit_balance_ratio"] = (df["current_credit_balance"] / df["current_loan_amount"]).fillna(0.0)
    
    return df

## data transformation

def rescale_data(df:pd.DataFrame, method:str='standardize', mode:str='training', columns:list=[], job_id:str="") -> pd.DataFrame:
    """
    Rescale data.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str, optional): The rescaling method, either 'standardize' or 'minmax'. Default is 'standardize'.
        mode (str, optional): The mode of operation, either 'training' or 'inference'. Default is 'training'.
        columns (list, optional): The list of columns to rescale. Default is an empty list.
        job_id (str, optional): The job ID. Default is an empty string.
        
    Returns:
        pd.DataFrame: The rescaled DataFrame.
    """
    
    assert method in ('standardize', 'minmax'), f"{method} is not a valid method (standardize, minmax)"
    assert mode in ('training', 'inference'), f"{mode} is not a valid mode (training, inference)"
    for col in columns:
        assert col in df.columns

    if mode=='training':
        if method=='standardize':
            scaler = StandardScaler()
            scaler.fit(df[columns])
        if method=='minmax':
            scaler = MinMaxScaler()
            scaler.fit(df[columns])
        model = {
            'scaler': scaler,
            'method': method,
        }

        helpers.save_model_as_pickle(model, f"{config.PATH_DIR_MODELS}/{job_id}_numerical_scaler.pkl")
        df[list(map(lambda x: f"{method}_{x}", columns))] = scaler.transform(df[columns])
        return df
    if mode=='inference':
        model = helpers.load_model_from_pickle(model_name=f"{job_id}_numerical_scaler.pkl")
        scaler = model['scaler']
        method = model['method']
        for col in columns:
            try:
                df[col].astype(float)
            except:
                print("[DEBUG] Column skipped:", col)
        df[list(map(lambda x: f"{method}_{x}", columns))] = scaler.transform(df[columns])
        return df
    
## Preprocess
def split_train_test(df:pd.DataFrame, test_size:float, method:str='time based'):
    """
    Split data into train and test.
    :param df: DataFrame
    :param test_size: float, between 0 and 0.99
    :param method: str, 'time based' or 'random'
    :return: (DataFrame, DataFrame)
    """
    if method=='random':
        return df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[:int(len(df)*test_size)], df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[int(len(df)*test_size):]
    if method=='time based':
        unique_dates = sorted(df["application_date"].unique())
        train_dates = unique_dates[:int(len(unique_dates)*(1-test_size))]
        test_dates = unique_dates[unique_dates.index(train_dates[-1])+1:]
        train_df = df[df["application_date"].isin(train_dates)]
        test_df = df[df["application_date"].isin(test_dates)]

        return train_df, test_df
    
    raise(ValueError(f"{method} is not a valid method (time based, random)"))

def preprocess_data(df:pd.DataFrame, mode:str, job_id:str=None, rescale=False, ref_job_id:str=None) -> pd.DataFrame:
    """
    Pre-process data and save preprocessed datasets for later use.
    :param df: DataFrame
    :param mode: str, 'training' or 'inference'
    :param job_id: str, job_id for the preprocessed dataset
    :param rescale: bool, whether to rescale data.
    :param ref_job_id: str, job_id of the last deployed model. Usefull when doing inference.
    :return: DataFrame
    """
    assert mode in ('training', 'inference')
    
    if mode=='training':
        assert config.TARGET in df.columns, f"{config.TARGET} not in {df.columns}"

    df.columns = list(map(str.lower, df.columns))
    initial_size = df.shape[0]
    df = df[df["customer_id"].notnull() & df["loan_id"].notnull() & df["loan_status"].notnull()]
    if mode=='training':
        df["loan_status"] = df["loan_status"].str.lower()
    if df.shape[0] != initial_size:
        print(f"[WARNING] Dropped {initial_size - df.shape[0]} rows with null values in (customer_id, loan_id, loan_status)")
    
    df = enforce_datatypes(df, cat_vars=cat_vars, num_vars=num_vars)
    
    df = engineer_features(df)
    
    if mode=='training':
        # split train and test data before encoding categorical variables and imputing missing values
        train_df, test_df = split_train_test(df, config.TEST_SPLIT_SIZE, method=config.SPLIT_METHOD)
        train_df = encode_categorical_variables(train_df, mode="training", purpose_encode_method=config.PURPOSE_ENCODING_METHOD, job_id=job_id)
        train_df = impute_missing_values(train_df, method="basic", mode="training", job_id=job_id)
        if rescale:
            train_df = rescale_data(train_df, method=config.RESCALE_METHOD, mode="training", columns=num_vars + engineered_vars["numerical"])
        helpers.save_dataset(train_df, os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_training.csv"))
        preprocess_data(test_df, mode="inference", job_id=job_id, ref_job_id=job_id)
    else:
        # if mode is infer, no need to split train and test data
        test_df = encode_categorical_variables(df, mode="inference", purpose_encode_method=config.PURPOSE_ENCODING_METHOD, job_id=ref_job_id)
        test_df = impute_missing_values(test_df, method="basic", mode="inference", job_id=ref_job_id)
        if rescale:
            test_df = rescale_data(test_df, method=config.RESCALE_METHOD, mode="inference", columns=num_vars + engineered_vars["numerical"])
        helpers.save_dataset(test_df, os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_inference.csv"))
    return test_df
        