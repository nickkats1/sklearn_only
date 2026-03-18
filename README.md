# Sklearn Only

This is a repo from my old github account that was later merged with this github account that only consists of scikit-learn models. 

## Rules
- Rule #1: Every project must only contain scikit-learn models.
- Rule #2: No NLP projects.
- Rule #3: Every project must be a pipeline (data → preprocessing → model → evaluation → optionally deployment).


## Note
This repository is no longer actively maintained. I will leave this up with faults and all.


### Project Structure

```
├── airplane # Multiple Regression models
│   ├── artifacts
│   ├── config.yaml
│   ├── helpers
│   ├── images
│   ├── logs
│   ├── main.py
│   ├── notebooks
│   ├── requirements.txt
│   ├── src
│   ├── styles
│   ├── templates
│   ├── tests
│   ├── Untitled-1.ipynb
│   └── venv
├── creditrisk # multiple classification models
│   ├── config.yaml
│   ├── helpers
│   ├── logs
│   ├── notebooks
│   ├── src
│   └── venv
├── hmda        # Loan approval classification (This was the first classification dataset I touched years ago.)
│   ├── app.py
│   ├── artifacts
│   ├── config.yaml
│   ├── data
│   ├── Dockerfile
│   ├── helpers
│   ├── images
│   ├── logs
│   ├── main.py
│   ├── notebooks
│   ├── requirements.txt
│   ├── src
│   ├── static
│   ├── templates
│   ├── tests
│   └── venv
├── houseprice      # multiple regression models (This is the first dataset I touched years ago.)
│   ├── app.py
│   ├── artifacts
│   ├── config.yaml
│   ├── data
│   ├── Dockerfile
│   ├── helpers
│   ├── images
│   ├── logs
│   ├── main.py
│   ├── setup.py
│   ├── src
│   ├── static
│   ├── templates
│   └── venv
├── README.md
└── utilization    # Multiple regression models
    ├── artifacts
    ├── config.yaml
    ├── helpers
    ├── images
    ├── logs
    ├── main.py
    ├── notebooks
    ├── requirements.txt
    ├── src
    └── venv
```


## Projects

### airplane
- Focus: multiple regression experiments.
- Entry points: `main.py` and a set of notebooks.
- Model family: scikit-learn regressors (see `src/`).

### creditrisk
- Focus: multiple classification experiments (credit scoring / risk).
- Entry points: `main.py` / notebooks in `notebooks/`.
- Model family: scikit-learn classifiers.

### hmda
- Focus: loan approval classification.
- Contains a `Dockerfile` and `app.py` (small API / demo UI).
- Artifacts and `main.py` for training/evaluation.

### houseprice
- Focus: house price regression experiments, packaged with `setup.py`.
- Contains `Dockerfile` and `app.py` for demo UI or API.
- See `src/` for preprocessing and model pipeline code.

### utilization
- Focus: multiple regression tasks around utilization forecasting.
- Entry point: `main.py` and supporting `src/` modules.







