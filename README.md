# dynamic_risk_assesment_system
Creating and deploying a model that can assess the attrition risk of a company's clients.  
The project encompasses the complete ML model deployment process, including continuous data ingestion and monitoring for model drift, necessitating retraining or redeployment decisions. It exposes the model via an API with multiple endpoints, offering inference and various statistics on model performance, data, and pipeline health.

# Introduction
The primary goal of the project is to construct a classifier capable of predicting customer churn and conducting comprehensive risk assessments. Synthetic datasets are furnished for training and continuous operation simulation. The focus lies on the MLOps components rather than the model's quality.

# process overview
The workflow is segmented into distinct components, including:

- Data ingestion
- Model training (utilizing the logisticRegression sklearn model for binary classification)
- Model scoring
- Deployment of pipeline artifacts into production
- Model monitoring, reporting, and statistics
- API setup for ML diagnostics and results
- Process automation with data ingestion and model drift detection via CRON job
- Retraining/redeployment in response to model drift

# Step to reproduce
The project is deployed on a Windows Python environment utilizing WSL2 Linux. The API is implemented using the Flask framework. It's essential to launch the model API prior to executing other project components, which can be achieved by running the app.py script. This script instantiates various project API endpoints, including inference capabilities.

- ingestion.py: ingests new data, aggregates and prepares data for model training.
- training.py: trains of a logistic regression model from the given data.
- deployment.py: deploys production models from trained models.
- scoring.py: scores the model in a production environment and validates it with a test dataset.
- diagnostics.py: Generates analyses and diagnostics for the model developed.
- reporting.py: allows for the generation of confusion matrix plot for the model prediction. 
- apicalls.py: makes API calls to gather all diagnostics and information related to model performance.
- fullprocess.py: regularly executed using a CRON job. It monitors the availability of new data, checks for model drift, and decides whether to retrain and redeploy an updated model in case drift is detected.

# Cron job
- Check if cron job is active, else run a cron job inside windows subsystem for linux using `sudo service cron start` 
- create new cron job using `crontab -e`
- the cron job should call the app.py and fullprocess.py script every 15 minutes in order to automate the whole process from data ingestion to model redeployment whenever necessary. 