# Expense Categorisation ML Classifier

This machine-learning project classifies expense transactions automatically. It trains a scikit-learn pipeline on labelled historical data, using TF-IDF vectorisation and logistic regression, to predict categories based on description text. It provides a simple CLI to train the model and save it for later use.

## Features

- Load labelled expense data from CSV (Description, Amount, Vendor, Category).
- Vectorise text descriptions with TF-IDF.
- Train a logistic regression classifier with cross-validation.
- Evaluate performance and output classification report.
- Save the trained model using joblib for integration into other systems.
