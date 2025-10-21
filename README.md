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