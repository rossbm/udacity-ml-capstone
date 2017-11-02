udacity-ML-capstone
==============================

This github repo contains my final project for Udacity's Machine Learning Nanodegree. It contains my final report and all the necessary code to reporduce my results.

Setup Instructions
-------------
To download, please use the following command:
```
git clone https://github.com/wertu234/udacity-ml-capstone
```

This will download the repo to the current directory. To ensure that you have the necessary packages, please use the Anaconda distribution of Python. With Anaconda, you will have access to the conda environment manager. In the newly created folder, type:
```
conda env create -f environment.yml
```

This will create a new environment named _capstone_. Please note that it will download a bunch of packages. To activate the environment:
*  Windows: ```activate capstone```
*  macOs and Linux: ```source activate capstone```

As a final step, please intstall xgboost into this environment. Please see http://xgboost.readthedocs.io/en/latest/build.html#windows-binaries for OS specific instructions.

After this has been completed, you can activate Jupyter Notebook by typing ```jupyter notebook``` into your shell. You can then explore the notebooks. The notebooks are ordered. They can be run in or out of sequence. If you run a later notebook without having run an earlier one, the prerequisite files will be created.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources, either downloaded or scraped. More or less unprocessed. 
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                        and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project. Generally code that is called by the Jupyter notebooks
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │   
        │   
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
