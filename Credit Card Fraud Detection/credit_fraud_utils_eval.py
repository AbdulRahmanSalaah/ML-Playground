from sklearn.metrics import average_precision_score, confusion_matrix, classification_report, f1_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"🎯 F1-Score: {f1:.4f}")
    print(f"📈 PR-AUC:   {pr_auc:.4f}")

    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n🧮 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0 (Not Fraud)", "Actual 1 (Fraud)"],
        columns=["Predicted 0", "Predicted 1"]
    )
    print(cm_df)
    
def evaluate_training_model(model, X_train, y_train):
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_f1 = f1_score(y_train, y_train_pred)
    train_pr_auc = average_precision_score(y_train, y_train_prob)
    print(f"\n📊 [Train] F1-Score: {train_f1:.4f} | PR-AUC: {train_pr_auc:.4f}")