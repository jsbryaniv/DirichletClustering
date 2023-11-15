
# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from model import DirichletClustering


### Load Data ###

# Load data
data = np.genfromtxt('data/diabetes.csv', delimiter=',', skip_header=True)
data = data[:, 1:]                               # Ignore first column (# pregnancies)
data = data[~np.any(data[:, :-1] == 0, axis=1)]  # Filter our rows with missing values

# Split data into training and test sets
np.random.shuffle(data)
split = int(0.8 * data.shape[0])
data_train = data[:split, :]
data_test = data[split:, :]


### Run Model ###

# Run model
n_classes = 3
model = DirichletClustering(n_classes)
variables, samples = model.train(data_train)

# Predict classes on test set
data_test_masked = data_test.copy()
data_test_masked[:, -1] = np.nan
s_pred = model.predict(data_test_masked, variables)

# Compute outcomes
outcomes = np.zeros((n_classes, 2))
for i in range(data_test.shape[0]):
    s = s_pred[i]
    b = int(data_test[i, -1])
    outcomes[s, b] += 1
outcomes = 100 * outcomes / np.sum(outcomes, axis=1)[:, None]
predicted = np.zeros((n_classes, 2))
for i in range(n_classes):
    predicted[i, 0] = 100 * (1 - variables.x[i])
    predicted[i, 1] = 100 * variables.x[i]


### Plot Results ###

# Set up figure
fig, ax = plt.subplots(2, n_classes, sharex=True, sharey=True)
plt.ion()
plt.show()
ax[0, 0].set_ylabel('Predicted')
ax[1, 0].set_ylabel('Actual')

# Loop over classes
for i in range(n_classes):

    # Make pie chart
    pie = ax[0, i].pie(predicted[i, :])
    ax[1, i].pie(outcomes[i, :])  

    # Add legend to bottom of figure
    ax[0, i].set_title(f'Class {i}')
    ax[1, i].legend(
        pie[0],
        labels=[
            f'Healthy\nPredicted={int(predicted[i, 0])}%\nMeasured={int(outcomes[i, 0])}%',
            f'Diabetic\nPredicted={int(predicted[i, 1])}%\nMeasured={int(outcomes[i, 1])}%'
        ],
        loc='center',
        bbox_to_anchor=(0.5, -0.3),
    )       
plt.tight_layout()


# Done
print('Done.')
