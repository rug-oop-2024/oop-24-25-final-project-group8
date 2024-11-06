
# Assignment instructions

## Terminology 
Before you move on to implementing the requirements. It is important to understand the context of the problem you are dealing with. These are concepts taken from real industrial applications.

### Definitions
- **AutoML**: typically an industry software or platform used to help train models without having to code pipelines. (e.g., H2O, teapot, autoop).
- **Artifact**: an abstract object refering to a asset which is stored and includes information about this specific asset (e.g., datasets, models, pipeline outputs, etc.).
```json
{
    "asset_path": "users/mo-assaf/models/yolov8.pth",
    "version": "1.0.2", 
    "data": b"binary_state_data",
    "metadata": {
        "experiment_id": "exp-123fbdiashdb",
        "run_id": "run-12378yufdh89afd",
    },
    "type": "model:torch",
    "tags": ["computer_vision", "object_detection"]
}
```
such artifacts can describe models as seen above, or pipeline objects needed by pipelines such as parameters used in preprocessing (e.g., auto scalers, one-hot encoders, text encoders.). These are also files that can be versioned or contain information such as input mappings. 

The `id` of an asset is derived as follows:
```
id={base64(asset_path)}:{version}
```
This maintains the referential identity since an artifact refers to an asset stored in a certain location using the `asset_path`.

- **Metric**: a function that maps $(\text{observations}, \text{groundtruth}) \implies \cal{R}$ a real number. Typically averaged over all point-wise comparisons.
By considering a dataset of $n$ data points, $\hat{y}^{(i)}$ as the model prediction for the $i$-th data point, and $y^{(i)}$ as the corresponding ground truth, the formulas are:
  - $\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}[\hat{y}^{(i)}=y^{i}]$, where $\mathbb{I}$ is the indicator function, which equates to 1 if the condition in brackets $[\cdot]$ is true, 0 if false.
  - $\text{Mean Squared Error} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2$.
  - For other classification metrics, be careful that these metrics must be for generic multi-class classification tasks and not for binary classification (binary classification = 2 classes, multi-class classification = more than 2 classes). Metrics such as "plain" Precision and F-1 score are not suitable for multi-class classification tasks.
- **Dataset**: an artifact that represents tabular data (simplified as CSV). In practice, datasets are usually split between a **training set** (which is used to train one or more models) and a **test set** (which is used to evaluate the model's performance). The training and test set must not overlap. The training and testing split is usually operated randomly and controlled by an argument that controls the % of the dataset that goes to the training set.
- **Model**: a function that maps input features to a target feature (also known as *response*) derived from a set of observations. Can be either a classification or a regression task, as seen during the lectures and assignments. A model has a `parameters` attribute that allows it to be saved and loaded, restoring the state of the model. Contrary to the `Model` class saw during assignment 1, `parameters` include both strict parameters (those useful for prediction) and hyperparameters (those useful for training), similarly to what done in Scikit-learn.
- **Feature**: individual measurable property, in this case, describing a column in a CSV and a type of either `categorical` or `numerical`. 
- **Pipeline**: a state machine that orchestrates the different stages. (i.e., preprocessing, splitting, training, evaluation). Pipelines can evolve to be quite complex but in this assignment we simplify them.

### Assumptions

We take some assumptions to simplify this task:

1. Only `categorical` and `numerical` features.
2. Perfectly observable data: No `NaN` or missing values, so no need to account for how to treat these values.
3. Pipelines only include these stages: preprocessing, splitting the data, training, and evaluation.
4. Feature selection is done manually, there is no automatic feature selection.

# Part 0: Set up
The base structure of the assignment is as follows:
```
.
├── INSTRUCTIONS.md
├── README.md
├── app # This is the streamlit app
│   ├── Welcome.py
│   ├── core # streamlit app logic
│   │   └── system.py
│   └── pages
        ...
├── assets # Storage to save your examples
├── autoop # You python library for AutoML
│   ├── core # core logic of automl
│   │   ├── database.py
│   │   ├── ml
│   │   │   ├── artifact.py
│   │   │   ├── dataset.py
│   │   │   ├── feature.py
│   │   │   ├── metric.py
│   │   │   ├── model
│   │   │   │   ├── __init__.py
│   │   │   │   ├── classification
│   │   │   │   ├── model.py
│   │   │   │   └── regression
│   │   │   └── pipeline.py
│   │   └── storage.py
│   ├── functional # automl functions
│   │   ├── feature.py
│   │   └── preprocessing.py
│   └── tests # Test you modules
│       ├── main.py
        ...
└── requirements.txt
```
Make sure to update the `requirements.txt` regularly.
```
conda activate <your env>
pip install -r requirements.txt
```

Run the `streamlit` app as follows:
```bash
python -m streamlit run app/Welcome.py
```
Run your tests as follows:
```
python -m autoop.tests.main
```

You will notice that some parts of the library are already pre-implemented.
**You are not allowed to functionally modify these parts, but you can add, e.g., docstrings and type hints to have style checks pass**.

# Part I: The core library

In this section you will focus on building and testing the core functionality of your AutoML library.

## Requirements

**Remember to add type hints and docstrings**

Make sure after implementing your classes you pass the respective tests. You will have to read existing implementation to understand how you need to implement your classes which is quite common working in a team or a company.

# Part II: Building the streamlit app

In this part you will integrate the library by importing your implemented classes in streamlit pages. **Notice that Tutorial III is designed to give you some knowledge on the functioning of Streamlit**.
## Extra requirements

*Notice: correctly implementing the requirements up to this point will grant 6 points.*
- `ST/deployment/predict`: Once the user loads a pipeline, prompt them to provide a CSV on which they can perform predictions.

# Part III: Go beyond

There are many suggestions in the `README.md` to further extend this and make it more interesting and creative. We give up to 2 points of bonus based on the complexity of your additions and demostration of OOP concepts. 

# Final thoughts

We hope that you learn more about OOP and the ML industry through this simplified assignment. Please read the submission instructions carefuly in `README.md`.


# Glossary

Useful terminology that can be useful in the understanding of the assignment:


- **Categorical feature**: a feature that can take on one of a limited, and usually fixed, number of possible values, assigning each observation to a particular group or category.
- **Classification**: a type of task where the response is categorical.
- **Covariates**: the features that are used to predict the response. It is synonymous with "predictor".
- **Feature**: individual measurable property, in this case, describing a column in a CSV and a type of either categorical or numerical. It is synonymous with "variable" or "attribute", although we avoid using the latter to avoid confusion with OOP lexicon.
- **Fit**: the process of training a model on a given task. This equates to adjusting the parameters with a given criterion to minimize a loss function.
- **Ground truth**: the actual value of the response in a given observation. It is usually indicated with the symbol $y$ and is used to assess the quality of the predictions according to a given metric. It may not be present outside of training.
- **Hyperparameters**: the external variables of a model that are set before the training process. They are used to control the training process. *NOTE: in the context of this project, hyperparameters are considered part of the parameters themselves.*
- **Loss function**: a function that measures the difference between the predicted response and the ground truth. It is used to adjust the parameters of the model during the training process.
- **Metric**: a function that takes in predictions and ground truth and measures the quality of the predictions made by the model. It is used to assess the performance of the model on a given task.
- **Model**: a function that maps input features to a target feature derived from a set of observations. Can represent either a classification or a regression task, as seen during lectures and assignments. Models have a fit behavior that allows them to be trained on a given task and a predict behavior that allows them to make predictions on the same task.
- **Numerical feature**: a feature that can take on any numerical value within a potentiallypotentially infinite interval.
- **Observation**: a row in a dataset. It is synonymous with "data point" or "statistical unit".
- **One-hot encoding**: a method used to convert categorical features into numerical features. It creates a vector of length $c$ (where $c$ is the number of categories) with a 1 in the position corresponding to the category and 0s elsewhere. For instance, if a feature has four categories, an observation falling in the third category would be encoded as $(0, 0, 1, 0)$.
- **Parameters**: the internal variables of a model that modified in the training process. They are used to make predictions.
- **Predict**: the process of using a model to make predictions on a given task. This equates to using the parameters to map the covariates to the response, using the parameters to produce the response.
- **Prediction**: the output of the predict behavior. It is usually indicated with the symbol $\hat{y}$.
- **Regression**: a type of task where the response is numerical.
- **Response**: the feature that the model is trying to predict. It is synonymous with "target". In this project, the response is always a single feature.
- **Task**: a type of problem (i.e., combination of covariates and response) that the model is trying to solve. It can be either a classification or a regression task.
