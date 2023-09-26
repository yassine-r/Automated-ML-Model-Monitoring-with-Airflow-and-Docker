

import traceback
import pandas as pd
import os
from sklearn.ensemble import BaseEnsemble


from deepchecks.tabular import Dataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import WholeDatasetDrift, DataDuplicates, NewLabelTrainTest, TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureLabelCorrelationChange, ConflictingLabels, OutlierSampleDetection 
from deepchecks.tabular.checks import WeakSegmentsPerformance, RocReport, ConfusionMatrixReport, TrainTestPredictionDrift, CalibrationScore, BoostingOverfit

import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'dags', 'src'))

import helpers
import config
import preprocess



def check_data_quality(df: pd.DataFrame, predictors: list[str], target: str, job_id: str) -> dict:
    """
    Checks for data quality and saves a report in the results directory.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        predictors (list[str]): Predictors to check for drifts.
        target (str): Target variable to check for drifts.
        job_id (str): Job ID.
    
    Returns:
        dict: Dictionary containing the report and the boolean result.
    """
    # Filter features and categorical features based on the columns in the DataFrame
    features = [col for col in predictors if col in df.columns]
    cat_features = [col for col in config.CAT_VARS if col in df.columns]
    
    # Create a Dataset object with the filtered features and categorical features
    dataset = Dataset(df, label=target, features=features, cat_features=cat_features, datetime_name=config.DATETIME_VARS[0])
    
    # Create a Suite object for data quality checks
    data_quality_suite = Suite("data quality",
        DataDuplicates().add_condition_ratio_less_or_equal(0.3), #Checks for duplicate samples in the dataset
        ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(0), #Find samples which have the exact same features' values but different labels
        FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9), #Return the PPS (Predictive Power Score) of all features in relation to the label
        OutlierSampleDetection(outlier_score_threshold=0.7).add_condition_outlier_ratio_less_or_equal(0.1), #Detects outliers in a dataset using the LoOP algorithm
    )
    
    # Run the data quality suite on the dataset
    report = data_quality_suite.run(dataset)
    
    try:
        # Save the report as an HTML file
        report_path = f"{config.PATH_DIR_RESULTS}/reports/{job_id}_data_quality_report.html"
        report.save_as_html(report_path)
        print(f"[INFO] Data quality report saved as {report_path}")
    except FileNotFoundError as e:
        print(f"[WARNING][DRIFTS.SKIP_TRAIN] {traceback.format_exc()}")
    
    # Return the report and the boolean result
    return {"report": report, "retrain": report.passed()}


def check_data_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, predictors: list[str], target: str, job_id: str) -> dict:
    """
    Check for data drifts between two datasets and decide whether to retrain the model.
    A report will be saved in the results directory.
    
    Args:
        ref_df (pd.DataFrame): Reference dataset.
        cur_df (pd.DataFrame): Current dataset.
        predictors (list[str]): Predictors to check for drifts.
        target (str): Target variable to check for drifts.
        job_id (str): Job ID.
    
    Returns:
        dict: Dictionary containing the report and the boolean result.
    """
    # Filter features and categorical features based on the columns in the DataFrames
    ref_features = [col for col in predictors if col in ref_df.columns]
    cur_features = [col for col in predictors if col in cur_df.columns]
    ref_cat_features = [col for col in config.CAT_VARS if col in ref_df.columns]
    cur_cat_features = [col for col in config.CAT_VARS if col in cur_df.columns]
    
    # Create Dataset objects for the reference and current datasets
    ref_dataset = Dataset(ref_df, label=target, features=ref_features, cat_features=ref_cat_features, datetime_name=config.DATETIME_VARS[0])
    cur_dataset = Dataset(cur_df, label=target, features=cur_features, cat_features=cur_cat_features, datetime_name=config.DATETIME_VARS[0])
    
    # Create a Suite object for data drift checks
    data_drift_suite = Suite("data drift",
        NewLabelTrainTest(),
        WholeDatasetDrift().add_condition_overall_drift_value_less_than(0.01), #0.2
        FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than(0.05), #0.2
        TrainTestFeatureDrift().add_condition_drift_score_less_than(0.01), #0.1
        TrainTestLabelDrift().add_condition_drift_score_less_than(0.01) #0.1
    )
    
    # Run the data drift suite on the reference and current datasets
    report = data_drift_suite.run(ref_dataset, cur_dataset)
    
    # Determine whether to retrain based on the results of the checks
    retrain = (len(report.get_not_ran_checks()) > 0) or (len(report.get_not_passed_checks()) > 0)
    
    try:
        # Save the report as an HTML file
        report_path = f"{config.PATH_DIR_RESULTS}/reports/{job_id}_data_drift_report.html"
        report.save_as_html(report_path)
        print(f"[INFO] Data drift report saved as {report_path}")
    except Exception as e:
        print(f"[WARNING][DRIFTS.check_DATA_DRIFT] {traceback.format_exc()}")
    
    # Return the report and the boolean result
    return {"report": report, "retrain": retrain}


def check_model_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, model: BaseEnsemble, predictors: list[str], target: str, job_id: str) -> dict:
    """
    Using the same pre-trained model, compare drifts in predictions between two datasets and decide whether to retrain the model.
    A report will be saved in the results directory.
    
    Args:
        ref_df (pd.DataFrame): Reference dataset.
        cur_df (pd.DataFrame): Current dataset.
        model (BaseEnsemble): Pre-trained model. Only scikit-learn and xgboost models are supported.
        predictors (list[str]): Predictors to check for drifts.
        target (str): Target variable to check for drifts.
        job_id (str): Job ID.
    
    Returns:
        dict: Dictionary containing the report and the boolean result.
    """
    # Filter features and categorical features based on the columns in the DataFrames
    ref_features = [col for col in predictors if col in ref_df.columns]
    cur_features = [col for col in predictors if col in cur_df.columns]
    ref_cat_features = [col for col in config.CAT_VARS if col in ref_df.columns]
    cur_cat_features = [col for col in config.CAT_VARS if col in cur_df.columns]
    
    # Create Dataset objects for the reference and current datasets
    ref_dataset = Dataset(ref_df, label=target, features=ref_features, cat_features=ref_cat_features, datetime_name=config.DATETIME_VARS[0])
    cur_dataset = Dataset(cur_df, label=target, features=cur_features, cat_features=cur_cat_features, datetime_name=config.DATETIME_VARS[0])
    
    # Create a Suite object for model drift checks
    model_drift_suite = Suite("model drift",
        #For each class plots the ROC curve, calculate AUC score and displays the optimal threshold cutoff point.
        RocReport().add_condition_auc_greater_than(0.7), 
        #Calculate prediction drift between train dataset and test dataset, Cramer's V for categorical output and Earth Movers Distance for numerical output.
        TrainTestPredictionDrift().add_condition_drift_score_less_than(max_allowed_drift_score=0.1) 
        )
    
    # Run the model drift suite on the reference and current datasets using the pre-trained model
    report = model_drift_suite.run(ref_dataset, cur_dataset, model)
    
    # Determine whether to retrain based on the results of the checks
    retrain = (len(report.get_not_ran_checks()) > 0) or (len(report.get_not_passed_checks()) > 0)
    
    try:
        # Save the report as an HTML file
        report_path = f"{config.PATH_DIR_RESULTS}/reports/{job_id}_model_drift_report.html"
        report.save_as_html(report_path)
        print(f"[INFO] Model drift report saved as {report_path}")
    except Exception as e:
        print(f"[WARNING][DRIFTS.check_MODEL_DRIFT] {traceback.format_exc()}")
    
    # Return the report and the boolean result
    return {"report": report, "retrain": retrain}
