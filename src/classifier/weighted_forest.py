import numpy as np
import random

class Weighted_Forest:
    class Cell:
        class Gate:
            def __init__(self, used_features, distance_function, learning_rate=0.01, boundery=2.5):
                self.boundery = boundery
                self.used_features = used_features
                self.distance_function = distance_function
                self.learning_rate = learning_rate

                self.gate_vector = np.random.normal(size=(self.used_features.shape[0]))

                self._saved_features = None

            def forward(self, features):
                self._saved_features = features[self.used_features]
                distance = self.distance_function(self._saved_features, self.gate_vector)
                print(distance)
                return distance < self.boundery, distance
            
            def backward(self, right_decision):
                if self._saved_features is None:
                    raise Exception("Run backward pass without run forward.")

                if right_decision:
                    self.gate_vector = self.gate_vector + self.learning_rate * (self._saved_features - self.gate_vector)
                else:
                    self.gate_vector = self.gate_vector + self.learning_rate * (self._saved_features + self.gate_vector)

                ##print("G: ", self.gate_vector)

                
        class Decision:
            def __init__(self, used_features, num_classes, distance_function, learning_rate=0.05):
                self.used_features = used_features
                self.num_classes = num_classes
                self.distance_function = distance_function
                self.learning_rate = learning_rate

                self.decision_vector = np.random.normal(size=(self.num_classes, self.used_features.shape[0]))

                self._saved_features = None

            def forward(self, features):
                self._saved_features = features[self.used_features]
                
                class_distances = np.zeros(shape=self.num_classes)
                for i in range(self.num_classes):
                    class_distances[i] = self.distance_function(self.decision_vector[i], self._saved_features)

                class_distances = class_distances / np.sum(class_distances)
                self._predicted_class = np.argmin(class_distances)
                return self._predicted_class, class_distances
            
            def backward(self, right_decision):
                if self._saved_features is None:
                    raise Exception("Run backward pass without run forward.")
                                
                if right_decision:
                    self.decision_vector[self._predicted_class] = self.decision_vector[self._predicted_class] + self.learning_rate * (self._saved_features - self.decision_vector[self._predicted_class])
                else:
                    self.decision_vector[self._predicted_class] = self.decision_vector[self._predicted_class] + self.learning_rate * (self._saved_features + self.decision_vector[self._predicted_class])
                ##print("D: ", self.decision_vector[self._predicted_class])


        def __init__(self, num_features, num_classes, distance_function):
            self.num_features = num_features
            self.num_classes = num_classes

            split = np.arange(start=0, stop=self.num_features, step=1, dtype=np.int32)
            np.random.shuffle(split)
            gate_vector_size = random.randint(1,self.num_features-1)

            self.gate = Weighted_Forest.Cell.Gate(used_features=split[:gate_vector_size], distance_function=distance_function)
            self.decision = Weighted_Forest.Cell.Decision(used_features=split[gate_vector_size:], num_classes=self.num_classes, distance_function=distance_function)

        def forward(self, features):
            self._saved_take_decision, _ = self.gate.forward(features)

            if self._saved_take_decision:
                predicted_class, class_distance = self.decision.forward(features)
                return predicted_class, class_distance
            
            return None
        
        def backward(self, right_decision):
            if self._saved_take_decision:
                self.gate.backward(right_decision=right_decision)
                self.decision.backward(right_decision=right_decision)
                


    def __init__(self, num_features, num_classes, distance_function):
        if num_features < 2 or num_classes <2:
            raise Exception("Classifier needs at least two features and two classes.")

        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_function = distance_function
        self.cells = []
        for i in range(4):
            self.cells.append(Weighted_Forest.Cell(self.num_features, self.num_classes, distance_function=self.distance_function))

    def forward(self, features):
        if type(features) != np.ndarray:
            raise Exception("Only Numpy arrays excepted.")

        predictions = []
        for i, cell in enumerate(self.cells):
            pred = cell.forward(features)
            if pred is not None:
                predictions.append(pred[1]) 

        prediction = np.mean(np.array(predictions), axis=0)
        return np.argmin(prediction)

    def backward(self, right_decision_bool):
        for cell in self.cells:
            cell.backward(right_decision=right_decision_bool)

    def fit(self, X, y, epochs=1):
        accurays = np.zeros(shape=(epochs))
        for e in range(epochs):
            counter = 0
            correct = 0
            for idx in range(X.shape[0]):
                out = self.forward(X[idx])
                self.backward(out == y[idx])
                counter += 1
                if out == y[idx]:
                    correct += 1

            accurays[e] = correct/counter

        return(accurays)
    
    def predict(self, X):
        predictions = np.zeros(shape=(X.shape[0]))
        for idx in range(X.shape[0]):
            predictions[idx] = self.forward(X[idx])
        return predictions
