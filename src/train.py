from data.loader import load_training_data
from classifier.weighted_forest import WeightedForest
from lib.distance_functions import euclidean_distance
from data.preparation import convert_str_to_int, min_max_scale, undersample
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

SEED = 42

## Data loading
X, y = load_training_data(random_state=42)
X, y = X.to_numpy(), y.to_numpy()

## Prepare data
X = min_max_scale(X)
y = convert_str_to_int(y)
X, y = undersample(X, y, random_seed=SEED)

## Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=SEED)

## Fit Classifier
wf = WeightedForest(X.shape[1], 3, distance_function=euclidean_distance, accuracy_goal=0.8, initializer_low=0, initializer_high=1, random_seed=SEED)
wf.fit(X_train,y_train,10)

## Test Classifier
y_pred = wf.predict(X_test)
print(y_pred, y_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
