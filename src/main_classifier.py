from game.pong import main as run_interactive_pong
from classifier.weighted_forest import Weighted_Forest
from lib.distance_functions import euclidean_distance
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    ##run_interactive_pong()

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    wf = Weighted_Forest(4, 3, distance_function=euclidean_distance)
    accuracys = wf.fit(X_train,y_train,100)
    epochs = np.arange(accuracys.shape[0])

    correct = 0
    predictions = wf.predict(X_test)
    for idx in range(X_test.shape[0]):
        if predictions[idx] == y_test[idx]:
            correct += 1
    print("Accuracy: ", correct/X_test.shape[0])


    plt.plot(epochs, accuracys)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('WF Performance Training')

    plt.show()