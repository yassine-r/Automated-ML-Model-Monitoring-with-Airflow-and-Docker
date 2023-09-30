import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import roc_auc_score,  accuracy_score, f1_score, precision_score, recall_score, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import config
import helpers




def performance_report(y_true, y_pred, y_prob):
    """
    Generate performance report for a model.

    Parameters:
    y_true (np.array): An array containing the true values.
    y_pred (np.array): An array containing the predicted values.
    y_prob (np.array): An array containing the prediction probabilities.

    Returns:
    dict: A dictionary containing various performance metrics.
    """
    # Create an empty dictionary to store the performance report
    report = dict()

    # Calculate and store the dataset size
    report["dataset size"] = y_true.shape[0]

    # Calculate and store the positive rate
    report["positive rate"] = y_true.sum() / y_true.shape[0]

    # Calculate and store the accuracy
    report["accuracy"] = accuracy_score(y_true, y_pred)

    # Calculate and store the F1 score
    report["f1"] = f1_score(y_true, y_pred)

    # Calculate and store the precision
    report["precision"] = precision_score(y_true, y_pred)

    # Calculate and store the recall
    report["recall"] = recall_score(y_true, y_pred)

    # Calculate and store the AUC score
    report["auc"] = roc_auc_score(y_true, y_prob)

    # Return the performance report
    return report


def select_model(df: pd.DataFrame, metric: str = config.MODEL_PERFORMANCE_METRIC, model_names: list = ["rf", "gb"], performance_thresh: float = config.MODEL_PERFORMANCE_THRESHOLD, degradation_thresh: float = config.MODEL_DEGRADATION_THRESHOLD) -> str:
    """
    Select the best model based on their performance reports.

    Parameters:
    df (pd.DataFrame): The performance report DataFrame.
    metric (str): The metric to select the best model (default: config.MODEL_PERFORMANCE_METRIC).
    model_names (list): The list of model names to select from (default: ["rf", "gb"]).
    performance_thresh (float): The threshold for the performance (default: config.MODEL_PERFORMANCE_THRESHOLD).
    degradation_thresh (float): The threshold for degradation (default: config.MODEL_DEGRADATION_THRESHOLD).

    Returns:
    str: The name of the selected model.

    Raises:
    Exception: If no model is selected due to all models having performance below the threshold.
    """
    # Create an empty list to store model degradation performance
    degradation_performance = []

    # Iterate over each model
    for model in model_names:
        # Check if the model's performance is below the performance threshold
        if df.loc[metric, f"{model}_train"] < performance_thresh:
            continue

        # Calculate the degradation
        degradation = df.loc[metric, f"{model}_train"] - df.loc[metric, f"{model}_test"]

        # Check if the degradation is below the degradation threshold
        if degradation < degradation_thresh:
            degradation_performance.append((model, degradation))

    # Check if any model meets the selection criteria
    if len(degradation_performance) == 0:
        raise Exception("No model selected: all models have performance below the threshold. Possible overfitting.")

    # Return the model with the minimum degradation
    return min(degradation_performance, key=lambda x: x[1])[0]


def train(train_dataset_filename:str=None, test_dataset_filename:str=None, job_id="", rescale=False):
    """
    Train a model on the train dataset loaded from `train_dataset_filename` and test dataset loaded from `test_dataset_filename`.

    Parameters:
    train_dataset_filename (str): The filename of the train dataset (default: None).
    test_dataset_filename (str): The filename of the test dataset (default: None).
    job_id (str): The job ID (default: "").
    rescale (bool): If True, scaled numerical variables are used (default: False).

    Returns:
        dict
    """
    if train_dataset_filename==None:
        train_dataset_filename = os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_training.csv")
    if test_dataset_filename==None:
        test_dataset_filename = os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_inference.csv")
    tdf = helpers.load_dataset(train_dataset_filename)
    vdf = helpers.load_dataset(test_dataset_filename)
    helpers.check_dataset_sanity(tdf)
    helpers.check_dataset_sanity(vdf)
    
    predictors = config.PREDICTORS
    target = config.TARGET
    if rescale:
        for col in predictors:
            if f"{config.RESCALE_METHOD}_{col}" in tdf.columns:
                tdf[col] = tdf[f"{config.RESCALE_METHOD}_{col}"]
            if f"{config.RESCALE_METHOD}_{col}" in vdf.columns:
                vdf[col] = vdf[f"{config.RESCALE_METHOD}_{col}"]
        
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    X, Y = tdf[predictors], tdf[target]
    report = dict()
    models = dict()
    for cl, name in [(rf, "rf"), (gb, "gb")]:
        print("[INFO] Training model:", name)
        cl.fit(X, Y)
        t_pred = cl.predict(X)
        v_pred = cl.predict(vdf[predictors])
        t_prob = cl.predict_proba(X)[:, 1]
        v_prob = cl.predict_proba(vdf[predictors])[:, 1]
        report[f"{name}_train"] = performance_report(Y, t_pred, t_prob)
        report[f"{name}_test"] = performance_report(vdf[target], v_pred, v_prob)
        models[name] = cl
        
    model_name = select_model(pd.DataFrame(report), metric=config.MODEL_PERFORMANCE_METRIC, model_names=list(models.keys()))
    report["final_model"] = model_name
    helpers.save_model_as_pickle(models[model_name], f"{job_id}_{model_name}")
    helpers.save_model_as_json(report, f"{job_id}_train_report")
    return report


def pick_model_and_deploy(job_id, models, df, metric="auc", predictors=config.PREDICTORS, target=config.TARGET) -> str:
    """
    Among all `models`, select the model that performs best on `df` and mark it for deployment.

    Parameters:
    job_id (str): The ID of the job.
    models (list): A list of dictionaries representing the models.
    df (pd.DataFrame): The DataFrame on which the models are evaluated.
    metric (str): The metric used to evaluate the models (default: "auc").
    predictors (list): The list of predictor variables (default: config.PREDICTORS).
    target (str): The target variable (default: config.TARGET).

    Returns:
    str: The name of the selected model for deployment.
    """
    # Check if the columns in `predictors` are present in `df`
    cols = set(predictors).difference(set(df.columns))
    assert cols == set(), f"{cols} not in {df.columns}"

    # Initialize variables for tracking the best model
    score = 0
    m_idx = 0

    # Iterate over the models and evaluate their performance on `df`
    for i, model in enumerate(models):
        y_true = df[target]
        y_pred = model["model"].predict(df[predictors])
        y_prob = model["model"].predict_proba(df[predictors])[:, 1]
        r = performance_report(y_true, y_pred, y_prob)
        if r[metric] > score:
            score = r[metric]
            m_idx = i

    # Persist the deploy report for the selected model
    helpers.persist_deploy_report(job_id, models[m_idx]["model_name"])

    # Return the name of the selected model for deployment
    return models[m_idx]["model_name"]
