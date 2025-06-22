import argparse
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_training_model


def train_neural_network(X, y, output_path, scaler):
    print("🚀 Training Neural Network (MLPClassifier)...")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)
    model.fit(X, y)
    evaluate_training_model(model, X, y)
    save_model(model, output_path , scaler=scaler)
    

def train_logistic_regression(X, y, output_path, scaler):
    print("🚀 Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.threshold = 0.7
    model.fit(X, y)
    evaluate_training_model(model, X, y)
    save_model(model, output_path , scaler=scaler)


def train_random_forest(X, y, output_path, scaler):
    print("🚀 Training Random Forest...")
    model = RandomForestClassifier( random_state=42 , n_estimators=50)
    model.fit(X, y)
    evaluate_training_model(model, X, y)
    save_model(model, output_path , scaler=scaler)
    


def train_voting_classifier(X, y, output_path , scaler):
    print("🚀 Training Voting Classifier...")
    logreg = LogisticRegression(random_state=42 , max_iter=1000)
    logreg.threshold = 0.7
    rf = RandomForestClassifier( random_state=42 , n_estimators=50)
    model = VotingClassifier(estimators=[('logreg', logreg), ('rf', rf)], voting='soft')
    model.fit(X, y)
    evaluate_training_model(model, X, y)
    save_model(model, output_path, scaler)



def save_model(model, path , scaler=None):
    joblib.dump({
        "model": model,
        "scaler": scaler
    }, path)
    print(f"✅ Saved model to {path}")



def main(args):
        # Load and preprocess training data
    train_df = load_data(args.train_data)
    X_train, y_train = preprocess_data(train_df)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X_train)

    # # Apply SMOTE to balance training set
    # smote = SMOTE(random_state=42 , sampling_strategy=0.01)
    # X_train_resampled, y_train_resampled = smote.fit_resample(x_scaled, y_train)

     # Train & save models
    train_neural_network(x_scaled, y_train, f"{args.output_dir}/model_nn.pkl" , scaler) 
    train_logistic_regression(x_scaled, y_train, f"{args.output_dir}/model_logreg.pkl" , scaler)
    train_random_forest(x_scaled, y_train, f"{args.output_dir}/model_rf.pkl" , scaler)
    train_voting_classifier(x_scaled, y_train, f"{args.output_dir}/model_voting.pkl", scaler)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',  default='data/split/train.csv', help='Path to training CSV')
    parser.add_argument('--output_dir', default='saved_models', help='Directory to save trained models')
    args = parser.parse_args()
    main(args)