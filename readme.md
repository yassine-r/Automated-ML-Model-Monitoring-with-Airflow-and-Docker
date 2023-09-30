# Automated ML Model Monitoring with Airflow and Docker

### Note: This project requires the data to be injected in the folder dags/data/source.
Data is accessible within the "dags > data > source" directory in the format of "year_month_day.csv." The pipeline automatically selects the most recently added file for comparison and drift checks.

## 1. What's new?
ML pipeline monitoring using:
- Deepcheks
- Airflow
- Slack integration: alerts

## 2. Environment Setup
Here we will simply setup our environment using docker.
1. Make sure [docker](https://docs.docker.com/get-started/) and [docker-compose](https://docs.docker.com/get-started/08_using_compose/) are setup properly
2. Make a github repo an check in all the code which can be found [here](https://s3.amazonaws.com/projex.dezyre.com/ml-model-monitoring-using-apache-airflow-and-docker/materials/code.zip).
3. Clone the gitrepo: `git clone git@github.com:yassine-r/d.cd`
4. To proceed, make sure
- Docker can have access to at least 4GB of Memory on your system
- Navigate to `dags/src/config.py` and ensure `RUN_LOCAL` is set to `False`
5. While in the same home directory as `docker-compose.py` start docker-compose by issuing this command on you terminal: `docker-compose up`

This will take a couple of minutes to boot up all containers. To check if all containers are running properly, you can run `docker ps --all`. You should see a list of all containers in `healthy` status

## 3. How to reset environments
1. Delete all files under the following subdirectories. In case subdirectories do not exist (due to .gitignore) please create them

-  `dags/data/source/*`
-  `dags/data/collected`
-  `dags/data/preprocessed`
-  `dags/models`
-  `dags/results`

2. Truncate the `mljob` table
- `truncate mljob;`
  
# Monitoring Machine Learning Pipeline

## 1. Traditional machine learning model training pipeline
1. data gathering
2. data preprocessing
3. model training
4. model evaluation
5. model serving

## 2. The idea behind model training monitoring
1. data integrity
2. data drift
3. concept drift
4. comparative analysis of models
