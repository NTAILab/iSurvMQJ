# iSurvMQJ: Survival Analysis as Imprecise Classification with Trainable Kernels

iSurvMQJ is a collection of kernel-based models for survival analysis. It formulates survival prediction as an imprecise multiclass classification task by discretizing time into intervals, enabling the model to handle censored data and uncertainty in event times. The approach leverages trainable kernel mechanisms to learn flexible, attention-based representations of risk over time.

This framework is particularly suited for time-to-event modeling, which is essential in medical prognosis, reliability engineering, and risk modeling. The repository includes implementations of all model variants along with training utilities and evaluation scripts.

## Installation

To ensure compatibility and avoid conflicts, it is recommended to set up an isolated Python environment. You can use [conda](https://docs.anaconda.com/miniconda/) for this purpose.

To install `iSurvMQJ` in development mode after cloning the repository, follow these steps:

```bash
git clone https://github.com/NTAILab/iSurvMQJ.git
cd iSurvMQJ
pip install -e .
```

## Package Contents

The repository is organized as a flat package with all model implementation:

**`isurvmqj/`**

The main package directory, containing all imprecise survival model variants and shared components:

* **`isurvj.py`** – defines imprecise survival model that simultaneously learns both attention weights and class probabilities, enabling end-to-end optimization of the survival time distribution under uncertainty.
* **`isurvmq.py`** – defines imprecise survival models where attention weights are learned, while class probabilities are generated under specific conditions. The loss minimization strategy for the attention weights depends on the selected variant — either iSurvM or iSurvQ. Optionally, the probability distribution can be fine-tuned in a subsequent stage.
* **`alpha_net.py`** – contains the class AlphaNet used to train attention weights in all model variants. AlphaNet is a neural network module that computes attention weights between a set of keys and a query using learned embeddings. It forms the core of the attention mechanism across the iSurvMQJ models.
* **`__init__.py`** – aggregates all model variants for simplified import.

**`examples/`**

Contains example notebooks demonstrating how to train and evaluate each survival model variant on survival data.

## Usage

Example usage is provided in the `examples` directory, including a demonstration of the model's application to survival datasets.

To use the model for survival analysis, follow these steps:

1. Preprocess the dataset, ensuring it contains censored survival times (e.g., time-to-event data) in the format `(delta, time)` where:
   - `delta`: Censoring indicator (1 if the event occurred, 0 if the data is censored).
   - `time`: The observed survival time.
   The target variable `y` should be in the form of a structured NumPy array with the fields `delta` and `time`.
   The feature tensor `X` should be a NumPy array, categorical features should be encoded.

2. Define the required model using `iSurvJ`, `iSurvMQ`.
3. Train the model and evaluate performance metrics, such as the C-index or Integrated Brier Score, for model evaluation.

Here’s an example of using `iSurv*` models for survival analysis:

```python
from isurvmqj import (
    iSurvJ,
    iSurvMQ,
)

from sksurv.datasets import load_veterans_lung_cancer
from sklearn.model_selection import train_test_split
from sksurv.preprocessing import OneHotEncoder

X, y = load_veterans_lung_cancer()
X = OneHotEncoder().fit_transform(X)
X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

isurvm = iSurvMQ(method='mean', num_epoch=10, num_generation=100, num_epoch_M_pi_B=10)
isurvq = iSurvMQ(method='quantile', num_epoch=10, num_generation=100, num_epoch_M_pi_B=10)
isurvj = iSurvJ(num_epoch=200, lr=1e-2, dropout_rate=0.6, reg_alpha=2e-3, entropy_reg=2)

isurvm.fit(X_train, y_train)
isurvq.fit(X_train, y_train)
isurvj.fit(X_train, y_train)

preds_isurvm = isurvm.predict(X_test)
preds_isurvq = isurvq.predict(X_test)
preds_isurvj = isurvj.predict(X_test)

c_index_isurvm = isurvm.score(X_test, y_test)
c_index_isurvq = isurvq.score(X_test, y_test)
c_index_isurvj = isurvj.score(X_test, y_test)

ibs_isurvm = isurvm.count_ibs(X_test, y_train, y_test)
ibs_isurvq = isurvq.count_ibs(X_test, y_train, y_test)
ibs_isurvj = isurvj.count_ibs(X_test, y_train, y_test)

print(f'C-index (iSurvM): {c_index_isurvm}')
print(f'IBS (iSurvM): {ibs_isurvm}')

print(f'C-index (iSurvQ): {c_index_isurvq}')
print(f'IBS (iSurvQ): {ibs_isurvq}')

print(f'C-index (iSurvJ): {c_index_isurvj}')
print(f'IBS (iSurvJ): {ibs_isurvj}')
```

This will train the `iSurvJ`, `iSurvM` and `iSurvQ` models on Veterans dataset and provide predictions on test data.

## Citation

If you use this project in your research, please cite it as follows:

...will be later.