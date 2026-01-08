import numpy as np
import random
from classifier.base_classifier import BaseClassifier

class WeightedForest(BaseClassifier):
    class Cell:
        class Gate:
            def __init__(self, used_features, distance_function, learning_rate=0.01, boundery=2.5, initializer_low=-10, initializer_high=10, random_seed=42):
                self.boundery = boundery
                self.used_features = used_features
                self.distance_function = distance_function
                self.learning_rate = learning_rate

                np.random.seed(random_seed+1)
                self.gate_vector = np.random.uniform(initializer_low, initializer_high, size=(self.used_features.shape[0]))

                self.saved_features = []

            def forward(self, features):
                features = features[self.used_features]
                distance = self.distance_function(features, self.gate_vector)

                self.saved_features.append(features)
                return distance < self.boundery, distance
            
            def backward(self, right_decision, learning_decay=0.9):
                if not self.saved_features:
                    raise Exception("Run backward pass without run forward.")

                change_vector = np.zeros(shape=self.gate_vector.shape)
                for idx, features in enumerate(reversed(self.saved_features)):
                    change_vector += (learning_decay**(idx-1)) * (features - self.gate_vector)

                if right_decision:
                    self.gate_vector = self.gate_vector + self.learning_rate * change_vector
                else:
                    self.gate_vector = self.gate_vector - self.learning_rate * change_vector

                self.saved_features = []

                
        class Decision:
            def __init__(self, used_features, num_classes, distance_function, learning_rate=0.05, initializer_low=-10, initializer_high=10, random_seed=42):
                self.used_features = used_features
                self.distance_function = distance_function
                self.learning_rate = learning_rate

                np.random.seed(random_seed+2)
                self.decision_vector = np.random.uniform(initializer_low, initializer_high, size=(num_classes, self.used_features.shape[0]))

                self.saved_features = []

            def forward(self, features):
                features = features[self.used_features]

                class_distances = np.zeros(shape=self.decision_vector.shape[0])
                for i in range(class_distances.shape[0]):
                    class_distances[i] = self.distance_function(self.decision_vector[i], features)

                class_distances = class_distances / np.sum(class_distances)
                self.predicted_class = np.argmin(class_distances)

                self.saved_features.append([features, self.predicted_class])
                return self.predicted_class, class_distances
            
            def backward(self, right_decision, learning_decay=0.9):
                if not self.saved_features:
                    raise Exception("Run backward pass without run forward.")
                                
                change_vector = np.zeros(shape=self.decision_vector.shape)
                for idx, (features, predicted_class) in enumerate(reversed(self.saved_features)):
                    change_vector[predicted_class] = (learning_decay**(idx-1)) * (features - self.decision_vector[predicted_class])

                if right_decision:
                    self.decision_vector = self.decision_vector + self.learning_rate * change_vector
                else:
                    self.decision_vector = self.decision_vector - self.learning_rate * change_vector

                self.saved_features = []

        def __init__(self, num_features, num_classes, distance_function, learning_decay=0.9, initializer_low=-10, initializer_high=10, random_seed=42):
            self.num_features = num_features
            self.num_classes = num_classes
            self.learning_decay = learning_decay

            split = np.arange(start=0, stop=self.num_features, step=1, dtype=np.int32)
            np.random.seed(random_seed+2)
            np.random.shuffle(split)
            gate_vector_size = random.randint(1,self.num_features-1)

            self.gate = WeightedForest.Cell.Gate(used_features=split[:gate_vector_size], distance_function=distance_function, initializer_low=initializer_low, initializer_high=initializer_high)
            self.decision = WeightedForest.Cell.Decision(used_features=split[gate_vector_size:], num_classes=self.num_classes, distance_function=distance_function, initializer_low=initializer_low, initializer_high=initializer_high)

            self.made_decision = False ## Was once in the forward paths a decision taken

            self._record = [0,0,0]

        def forward(self, features):
            take_decision, _ = self.gate.forward(features)
            self._record[0] += 1 ## How often the cell got called

            if take_decision:
                self.made_decision = True
                predicted_class, class_distance = self.decision.forward(features)

                self._record[1] += 1 ## How often the cell took a decision
                return predicted_class, class_distance
            
            return None
        
        def backward(self, right_decision):
            if self.made_decision:
                if right_decision:
                    self._record[2] += len(self.gate.saved_features)     ## How often the cell made the right decision

                self.gate.backward(right_decision=right_decision, learning_decay=self.learning_decay)
                self.decision.backward(right_decision=right_decision, learning_decay=self.learning_decay)
                self.made_decision = False

        def get_lifetime(self) -> int:
            ## How often did the cell got called
            return self._record[0]
        
        def get_actiontime(self) -> int:
            ## How often did the cell got called and made an action/dicision
            return self._record[1]
        
        def get_righttime(self) -> int:
            ## How often was it the right decision
            return self._record[2]
        
        def get_gate_vector(self):
            return self.gate.gate_vector.copy()
        
        def get_gate_used_features(self):
            return self.gate.used_features.copy()

    def __init__(self, num_features, num_classes, distance_function, learning_decay=0.9, accuracy_goal=0.8, initializer_low=0, initializer_high=10, random_seed=42):
        if num_features < 2 or num_classes <2:
            raise Exception("Classifier needs at least two features and two classes.")

        self.num_features = num_features
        self.num_classes = num_classes
        self.distance_function = distance_function
        self.learning_decay = learning_decay
        self.accuracy_goal = accuracy_goal
        self.initializer_low = initializer_low
        self.initializer_high = initializer_high
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.cells = []
        for i in range(4):
            self.add_cell(random_seed=self.random_seed+i)

        self._record = [0,0]    ## Total, Right

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

    def backward(self, right_decision: bool):
        self._record[0] += 1
        if right_decision:
            self._record[1] += 1

        _new_cells = []
        for cell in self.cells:
            cell.backward(right_decision=right_decision)

            if cell.get_lifetime() > 150:
                if cell.get_actiontime() / cell.get_lifetime() < 0.005:
                    print("Removed Cell because of inactivity", cell.get_actiontime() / cell.get_lifetime())
                    continue
                if cell.get_righttime() / cell.get_actiontime() < 0.2:
                    print("Removed Cell because of bad decisions")
                    continue

                if cell.get_lifetime() > 1000:
                    if cell.get_righttime() / cell.get_actiontime() < self.accuracy_goal:
                        print("Removed Cell because of bad decisions")
                        continue

            _new_cells.append(cell)
        self.cells = _new_cells

        if self._record[1] / self._record[0] < self.accuracy_goal and self._record[0] > 1000:

            ## Remove similar cells
            remove_indexes = []
            for idx_a in range(len(self.cells)):
                for idx_b in range(idx_a+1, len(self.cells)):
                    if np.array_equal(self.cells[idx_a].get_gate_used_features(), self.cells[idx_b].get_gate_used_features()):
                        d = self.distance_function(self.cells[idx_a].get_gate_vector(), self.cells[idx_b].get_gate_vector())
                        if d < 2:
                            remove_indexes.append(idx_b)
                            print(f"Remove Cell {idx_b} because of simiarity")
            self.cells = [cell for i, cell in enumerate(self.cells) if i not in remove_indexes]

            ## Add Cells
            self.add_cell(random_seed=self.random_seed*len(self.cells))
            self._record[0] = 0
            self._record[1] = 0

    def fit(self, X, y, epochs=1):
        accurays = np.zeros(shape=(epochs))
        for idx_epoch, e in enumerate(range(epochs)):
            counter = 0
            correct = 0
            for idx in range(X.shape[0]):
                out = self.forward(X[idx])
                self.backward(out == y[idx])
                counter += 1
                if out == y[idx]:
                    correct += 1

            accurays[e] = correct/counter
            print(f"{idx_epoch+1} Num cells: {len(self.cells)}")

        return(accurays)
    
    def predict(self, X):
        predictions = np.zeros(shape=(X.shape[0]))
        for idx in range(X.shape[0]):
            predictions[idx] = self.forward(X[idx])
        return predictions

    def add_cell(self, random_seed=42):
        self.cells.append(WeightedForest.Cell(self.num_features, self.num_classes, distance_function=self.distance_function, learning_decay=self.learning_decay, initializer_low=self.initializer_low, initializer_high=self.initializer_high, random_seed=random_seed))