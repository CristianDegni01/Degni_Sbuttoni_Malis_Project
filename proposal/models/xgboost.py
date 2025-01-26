from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class XGBoostModel:
    # Initializes an XGBoost model with the given hyperparameters.
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=None, scale_pos_weight=None):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    # Trains the XGBoost model using the provided training data.
    def train(self, X_train, y_train):
        print("Starting XGBoost model training.")
        self.model.fit(X_train, y_train)
        print("XGBoost model trained successfully.")

    # Evaluates the model's performance on the given dataset and returns the accuracy score.
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Evaluation results:")
        print(classification_report(y, y_pred))
        return accuracy

    # Makes predictions using the trained XGBoost model.
    def predict(self, X):
        return self.model.predict(X)

    # Saves the trained XGBoost model to a file.
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"XGBoost model saved to {file_path}.")

    # Loads an XGBoost model from a file.
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        print(f"XGBoost model loaded from {file_path}.")