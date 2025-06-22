import argparse
import joblib
import os
from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model



def evaluate_single_model(model_path, X_test, y_test):
    print(f"\n📦 Evaluating model: {os.path.basename(model_path)}")
        # Load model object
    model_dict = joblib.load(model_path)
    model = model_dict["model"]
    scaler = model_dict.get("scaler", None)

    if scaler:
        print("🔄 Applying scaler to test data...")
        X_test = scaler.transform(X_test)
    else:
        print("⚠️ No scaler found in model, using raw test data.")
        
    # Evaluate the model
    evaluate_model(model, X_test, y_test)




def main(args):
    # Load and preprocess test data
    print(f"📄 Loading test data from: {args.test_data}")
    test_df = load_data(args.test_data)
    X_test, y_test = preprocess_data(test_df)  # no scaler
    

    # Evaluate each model
    for name in ["model_nn.pkl" , "model_logreg.pkl", "model_rf.pkl", "model_voting.pkl"]:
        model_path = os.path.join(args.models_dir, name)
        evaluate_single_model(model_path, X_test, y_test)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='data/split/test.csv', help='Path to test CSV')
    parser.add_argument('--models_dir', default='saved_models', help='Directory with model .pkl files')
    args = parser.parse_args()
    main(args)