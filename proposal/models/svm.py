from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

class SVMModel:
    # Initializes an SVM model with the given hyperparameters.
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=None, class_weight=None):
        self.model = SVC(
            kernel=kernel, 
            C=C, 
            gamma=gamma, 
            random_state=random_state,
            class_weight=class_weight
        )
    
    # Trains the SVM model using the provided training data.
    def train(self, X_train, y_train):
        print("Starting SVM model training.")
        self.model.fit(X_train, y_train)
        print("SVM model trained successfully.")
    
    # Evaluates the model's performance on the given dataset and returns the accuracy score.
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Evaluation results:")
        print(classification_report(y, y_pred))
        return accuracy
    
    # Makes predictions using the trained SVM model.
    def predict(self, X):
        return self.model.predict(X)
    
    # Saves the trained SVM model to a file.
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"SVM model saved to {file_path}.")
    
    # Loads an SVM model from a file.
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        print(f"SVM model loaded from {file_path}.")