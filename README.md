# README
Databricks test project to create a time series forecasting model.

## Database
https://www.kaggle.com/datasets/samuelcortinhas/time-series-practice-dataset?select=train.csv
The database contains time series data for practice purposes. It includes a training dataset (`train.csv`) that can be used for various time series analysis and forecasting tasks.

## Poetry project management
Project is built using [Poetry](https://python-poetry.org/) for dependency management and packaging.

Initialised with:
```bash
poetry init
```
after initial commit.

If the virtual environment is not activated the commands can be run in it by prefixing with `poetry run`, e.g.
```bash
poetry run <command>
```
When adding first dependency virtual environment is created automatically:
Creating virtualenv databricks-timeseries-ddx-2ipW-py3.12 in /home/marijo/.cache/pypoetry/virtualenvs
Visual Studio Code correctly recognises the poetry virtual environment and suggests to select it as interpreter.

Dependencies can be added with:
```bash
poetry add <package>
```
Dev dependencies can be added with:
```bash
poetry add --dev <package>
```

Version and dependencies are managed in `pyproject.toml` file and poetry.lock file. Poetry.lock ensures consistent installs across different environments and should be committed to version control in case we are building an application. In case of libraries it is safer to not commit it and let users resolve dependencies on their own.

To activate the virtual environment use:
```bash
poetry env activate
```
After activation the venv is not visible in the terminal prompt.

## Databricks
To connect to Databricks workspace use in a notebook or python filess:
```python
mlflow.login()
```
Interactive authentication input will pop up in VSCode that will ask for Databricks host URL and token.
Token can be created in Databricks workspace by navigating to User Settings -> Access Tokens -> Generate New Token.  
Url is databricks workspace url, e.g. `https://<your-workspace>.cloud.databricks.com`.  
After successful login the credentials are stored securely in local profile file and can be used in future sessions without re-authentication.

### Environment variables
#### Local development
Environment variables are stored in `.env` file that is not committed to version control for security reasons.  
A sample file `.env-sample` is provided that contains the structure of the `.env` file.  
Copy the sample file to `.env` and fill in the required values before running
Load environment variables in notebooks or python files with:
```python
from dotenv import load_dotenv
load_dotenv()
```
#### Shared / production environment
Environment file pattern should not be used in shared or production environments. Use databricks secrets:  
https://medium.com/@generative_ai/environment-variables-setting-in-databricks-dde16e3c3888

### Dependency management
https://mlflow.org/docs/latest/ml/model/dependencies/#saving-code-dependencies  
When logging any model the dependencies are inferred from the notebook or can be added manually. When additional python scripts with own dependencies (within the same folder) are used we can use log_model() parameter `infer_code_paths=True` to include dependencies from those scripts as well.  
When including whole scripts folder we can use `code_paths=["./my_scripts.py"]` parameter to extract dependencies from the script.

## Performance metrics
Logging and publishing 70 ARIMA models to Databricks MLflow tracking server took around 20 minutes. Training the models themselves takes 40 seconds.

## Problems
### ARIMA model fails to log to Databricks with error:
```
Failed to train model for store 0, product 0: 'ARIMA' object has no attribute 'save'
```
Models of different libraries are saved in different ways:
https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018

MLFlow has support for different libraries - different log methods.

But also libraries save models in different ways:  
In statmodels we save the result of fit() method:
```python
model = ARIMA(endog=train_data, order=(p,d,q))
fitted_model = model.fit()
fitted_model.save("arima_model.pkl")
```
In scikit-learn we save the model object itself:
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
import joblib
joblib.dump(model, 'rf_model.pkl')
```

Backup solution is to create model like a script (Models from code):
https://www.mlflow.org/docs/latest/ml/model/models-from-code/

### We can not register our custom model in Model Registry (because of incorrect name)

It is possible even in Community edition:  
* Registering models can be done at the same time when logging the model with log_model() function.
* Model Registry URI is the same as tracking URI, no need to set it separately.
* log_model() function has to contain parameter `registered_model_name` so that the model can be registered while logging it.
* Registered model name has to have a qualified name - <workspace>.<schema>.<model_name> and by default workspace name is "workspace" and schema is "default". So the full name is "workspace.public.<model_name>".

### MLflow model logging does not capture all dependencies from custom python model
When the crash below happens the dependencies will fallback to 'cloudpickle==3.1.1' and mlflow and the model will fail to start when served due to missing dependencies.

```
2025/11/02 21:51:10 WARNING mlflow.utils.environment: Encountered an unexpected error while inferring pip requirements (model URI: /tmp/tmpq5q_33ck/model, flavor: python_function). Fall back to return ['cloudpickle==3.1.1']. Set logging level to DEBUG to see the full traceback. 
2025/11/02 21:51:10 DEBUG mlflow.utils.environment: 
Traceback (most recent call last):
  File "/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages/mlflow/utils/environment.py", line 434, in infer_pip_requirements
    return _infer_requirements(
           ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages/mlflow/utils/requirements_utils.py", line 495, in _infer_requirements
    modules = _capture_imported_modules(model_uri, flavor, extra_env_vars=extra_env_vars)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages/mlflow/utils/requirements_utils.py", line 391, in _capture_imported_modules
    _run_command(
  File "/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages/mlflow/utils/requirements_utils.py", line 265, in _run_command
    raise MlflowException(msg)
mlflow.exceptions.MlflowException: Encountered an unexpected error while running ['/mnt/e/Projekti/ml/databricks-timeseries/.venv/bin/python', '/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages/mlflow/utils/_capture_modules.py', '--model-path', '/tmp/tmpq5q_33ck/model', '--flavor', 'python_function', '--output-file', '/tmp/tmppln60pap/imported_modules.txt', '--error-file', '/tmp/tmppln60pap/error.txt', '--sys-path', '["/tmp/tmplas0wwjp/model/code", "/tmp/tmpgv63ggk5/model/code", "/usr/lib/python312.zip", "/usr/lib/python3.12", "/usr/lib/python3.12/lib-dynload", "", "/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages"]']
exit status: -9
```
For posterity purposes the command that hangs looks like this:
```bash
poetry run dotenv run -- /mnt/e/Projekti/ml/databricks-timeseries/.venv/bin/python \
  /mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages/mlflow/utils/_capture_modules.py \
  --model-path /mnt/e/Projekti/ml/databricks-timeseries/models/helpful-agent-no-memory-model \
  --flavor python_function \
  --output-file /tmp/tmpva4anlv7/imported_modules.txt \
  --error-file /tmp/tmpva4anlv7/error.txt \
  --sys-path '["/usr/lib/python312.zip", "/usr/lib/python3.12", "/usr/lib/python3.12/lib-dynload", "", "/mnt/e/Projekti/ml/databricks-timeseries/.venv/lib/python3.12/site-packages"]'
```
Possible solutions:
https://mlflow.org/docs/latest/ml/model/dependencies/   
* possible to extend timeout by ssetting environment variable `MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT` to higher value (default is 120 seconds). This solution didnt help even when extended to 20 minutes
* possible that dependency inference fails because there is some mismatch in dependency versions. We can to use UV to lock down dependencies by specifying environment variable `MLFLOW_LOCK_MODEL_DEPENDENCIES=true`. This worked once but couldnt be reproduced later and even then some dependencies were missing.
* Specify dependencies manually:
  * Install poetry export plugin: `poetry self add poetry-plugin-export`
  * Export dependencies to requirements.txt: `poetry export -f requirements.txt --output requirements.txt --without-hashes`
  * When logging the model specify parameter `pip_requirements="requirements.txt"` and use requirements.txt in log_model() function. This solution is confirmed to work consistently.