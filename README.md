# Vector-valued Kernel Ridge Regression

This is a `sklearn`-compatible implementation of a Kernel Ridge Regression model for predicting in multiple dimensions.

### Installation
After cloning the repository install requirements with
```
python3.9 -m pip install -r requirements.txt
```

### Data

Get the data at [this link](https://www.icloud.com/iclouddrive/0jvQXQBjLjRewMmv2at2oyWCg#dipole-data) and unpack it into the repository folder.

### Running experiments

Four experiments evaluating the model's performance are given in the `experiments` folder:
* `descriptor` compares using the Coulomb data descriptor vs. no descriptor at all
* `similarity` compares the Matérn and Gauss similarity measures
* `schnet` compares with the SchNet [[1]](https://github.com/atomistic-machine-learning/SchNet) deep model
* `molecules` compares performance across different molecule sizes

To run experiments use e.g.

```
python3.9 -m experiments.similarity
```

The resulting learning curves will be placed in the `experiments/results` folder.

---
The `notebooks` folder contains jupyter notebooks in part deriving the implementaition of KRR for energy and then for dipole moment prediction, as well as one exploring the model's errors.
