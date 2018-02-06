udacity-ML-capstone
==============================

This github repo contains my final project for Udacity's Machine Learning Nanodegree. It contains my final report and all the necessary code to reproduce my results. I have setup a live demo of the final model on my personal website: http://www.beck-macneil.ca/evaluate-joke. Give it whirl!

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

This will create a new environment named _capstone_env_. Please note that it will download a bunch of packages. To activate the environment:
*  Windows: ```activate capstone_env```
*  macOs and Linux: ```source activate capstone_env```

After this has been completed, you can activate Jupyter Notebook by typing ```jupyter notebook``` into your shell. You can then explore the notebooks. The notebooks are ordered and should be run in sequence, more or less. If you run a later notebook without having run an earlier one, it is possible that an exception will be raised since some notebooks rely on the outputs of previous notebooks.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources, either downloaded or scraped. More or less unprocessed. 
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── models             <- Trained and serialized models.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                        and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
	│
    ├── reports            <- Final report and proejct proposal.
    │   └── figures        <- Generated graphics and figures that have been incorporated into the final report.
    │
    ├── environment.yml   <- The environment file for reproducing the analysis environment, e.g.
    │                         generated with `conda export > environment.yml`
    │
    └── src                <- Python functions and classes that are used by the Jupyter notebooks. Trying to follow DRY (Don't repeat yourself) principal

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
