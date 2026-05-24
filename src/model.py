import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    classes_weights = list(y_train.value_counts())
    ratio = classes_weights[0] / classes_weights[1]
    
    model = XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("-" * 40)

def save_model(model, filename="churn_model.pkl"):
    # Create a models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)
    joblib.dump(model, filepath)
    print(f"💾 Model saved successfully to: {filepath}")