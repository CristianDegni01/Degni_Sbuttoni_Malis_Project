from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class RandomForestModel:
    
    # A wrapper class for training, evaluating, and saving a Random Forest model.
    
    # Initializes the Random Forest model with given hyperparameters.
    def __init__(self, n_estimators=100, max_depth=None, random_state=None, class_weight=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            class_weight=class_weight
        )
    
    # Trains the Random Forest model on the provided training data.
    def train(self, X_train, y_train):
        print("Starting training...")
        self.model.fit(X_train, y_train)
        print("Model successfully trained.")
    
    # Evaluates the model on the given dataset and prints classification metrics.
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Evaluation results:")
        print(classification_report(y, y_pred))
        return accuracy
    
    # Makes predictions on the given dataset.
    def predict(self, X):
        return self.model.predict(X)
    
    # Saves the trained model to a file.
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}.")
    
    # Loads a trained model from a file.
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}.")