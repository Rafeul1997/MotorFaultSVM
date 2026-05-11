# MotorFaultSVM

![Python](https://img.shields.io/badge/Python-3.9-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Machine Learning](https://img.shields.io/badge/ML-SVM-orange)

Python package for induction motor fault detection using Support Vector Machine.

## Features

- CSV dataset support
- Automatic preprocessing
- Feature scaling
- SVM classification
- Fault prediction

## Installation

```bash
pip install motorfaultsvm
```

## Example

```python
from motorfaultsvm import train_from_csv
from motorfaultsvm import predict_fault

model, scaler, accuracy = train_from_csv("im_data.csv")

print(accuracy)

sample = [45, 0.08, 3.1, 1420]

result = predict_fault(model, scaler, sample)

print(result)
```
