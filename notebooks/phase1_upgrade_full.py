import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import os
import pandas as pd

# Resolve data folder relative to this script
BASE_DIR = os.path.dirname(os.path.realpath(__file__))       # notebooks/
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

# Now load
print("📥 Loading drone dataset from", DATA_DIR)
drone_data = pd.read_csv(os.path.join(DATA_DIR, 'drone_data.csv'))

# ----- Feature Engineering -----
print("🚀 Cleaning columns and applying one-hot encoding...")
# Clean column names
drone_data.columns = drone_data.columns.str.strip().str.lower()

print("✅ Columns present:", list(drone_data.columns))

# Confirm required columns exist
required_cols = ['confidence_score', 'object_detected', 'is_threat']
for col in required_cols:
    if col not in drone_data.columns:
        raise KeyError(f"Required column '{col}' not found. Available: {list(drone_data.columns)}")

# One-hot encode 'object_detected'
object_dummies = pd.get_dummies(drone_data['object_detected'], prefix='obj', drop_first=True)

# Combine features
X = pd.concat([drone_data[['confidence_score']], object_dummies], axis=1)
y = drone_data['is_threat']

print("✅ Feature set shape:", X.shape)
print("✅ Feature preview:\n", X.head())


# ----- Train/Test Split -----
print("🔀 Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----- Model Training -----
print("⚙️ Starting GridSearchCV for RandomForestClassifier...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# ----- Evaluation -----
print("\n🏆 Model Evaluation:")
print(classification_report(y_test, y_pred))
print("✅ Best Parameters:", grid_search.best_params_)
print("✅ ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("✅ F1 Score:", f1_score(y_test, y_pred))

# ----- Feature Importance Plot -----
print("📊 Generating feature importance plot...")
plt.figure(figsize=(10, 6))
plt.barh(X.columns, best_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Drone Threat Classification")
plt.tight_layout()
feature_plot_path = '../dashboard/feature_importance_phase1.png'
plt.savefig(feature_plot_path)
plt.show()
print(f"✅ Feature importance plot saved to {feature_plot_path}")

# ----- Save Model -----
model_path = '../models/drone_threat_model_phase1.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Model saved to {model_path}")

# ----- Save Predictions -----
predictions_path = '../dashboard/drone_threat_predictions_phase1.csv'
pd.DataFrame({'actual': y_test.reset_index(drop=True),
              'predicted': y_pred}).to_csv(predictions_path, index=False)
print(f"✅ Predictions saved to {predictions_path}")
