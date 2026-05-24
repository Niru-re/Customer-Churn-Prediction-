from src.data_preprocessing import load_and_clean_data, prepare_datasets
from src.model import train_random_forest, train_xgboost, evaluate_model, save_model
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'telco_customer_churn.csv')
    
    print(f"Looking for dataset at: {os.path.abspath(data_path)}")
    
    if not os.path.exists(data_path):
        print("\n❌ ERROR: 'telco_customer_churn.csv' was NOT found!")
        return

    print("\n✅ Dataset found! Step 1: Loading and Preprocessing Data...")
    df = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = prepare_datasets(df)
    
    print("Step 2: Training Models...")
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    print("Step 3: Evaluating Models...\n")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    print("Step 4: Saving the Best Model...")
    # Saving XGBoost since it gave us better Churn recall performance
    save_model(xgb_model, "xgb_churn_model.pkl")

if __name__ == "__main__":
    main()