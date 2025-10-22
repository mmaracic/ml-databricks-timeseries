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

## Problems
* ARIMA model fails to log to Databricks with error:
```
Failed to train model for store 0, product 0: 'ARIMA' object has no attribute 'save'
```
Possible solution is to create model like a script (Models from code):
https://www.mlflow.org/docs/latest/ml/model/models-from-code/
