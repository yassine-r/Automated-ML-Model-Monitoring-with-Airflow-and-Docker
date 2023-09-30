import os
import json
import pickle
import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'dags', 'src'))

import helpers
import config
import preprocess
import etl


def load_model():
    if not os.path.isfile(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json")):
        json.dump({"prediction_model": None}, open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "w"))
    filename = json.load(open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "r"))["prediction_model"]
    with open(os.path.join(config.PATH_DIR_MODELS, filename), "rb") as f:
        model = pickle.load(f)
    return model


def batch_inference(job_id:str, ref_job_id:str, predictors=[]) -> dict:
    """
    :param job_id: str
    :param ref_job_id: str
    :param start_date: datetime.date
    :param end_date: datetime.date
    :param predictors: list
    :return: dict
    """
    model_type = helpers.get_model_type(ref_job_id)
    model = helpers.load_model_from_pickle(f"{ref_job_id}_{model_type}")
    collected_data = etl.collect_data(job_id=job_id)
    df = helpers.load_dataset(collected_data)
    test_df = preprocess.preprocess_data(df, mode="inference", job_id=job_id, ref_job_id=ref_job_id, rescale=False)
    test_df['prediction'] = model.predict(test_df[predictors])
    test_df['prediction'] = test_df['prediction'].apply(lambda x: "loan given" if x==1 else "loan refused")
    return dict(test_df[['loan_id', 'prediction']].values)


if __name__=='__main__':
    predictors = config.PREDICTORS
    job_id = helpers.generate_uuid()
    ref_job_id = helpers.get_latest_deployed_job_id(status="pass")
    preds = batch_inference(job_id, ref_job_id, predictors=predictors)
    print(preds)