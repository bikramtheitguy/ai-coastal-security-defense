import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

# ----- 1. Load Dataset -----
print("📥 Loading vessel dataset...")
v = pd.read_csv('data/vessel_data.csv')

# ----- 2. Clean & Check Columns -----
v.columns = v.columns.str.strip().str.lower()
print("✅ Columns present:", list(v.columns))

# ensure required columns exist
required = {'speed_knots', 'latitude', 'longitude', 'is_anomalous'}
missing = required - set(v.columns)
if missing:
    raise KeyError(f"Missing columns {missing}. Available: {list(v.columns)}")

# ----- 3. Feature & Target -----
X = v[['speed_knots']]
y = v['is_anomalous']

# ----- 4. Model Tuning -----
print("⚙️  Tuning IsolationForest via GridSearchCV...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'contamination': [0.05, 0.1]
}
iso_gs = GridSearchCV(
    IsolationForest(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',   # using the anomaly_flag to guide search
    n_jobs=-1
)
iso_gs.fit(X, y)
model = iso_gs.best_estimator_
print("✅ Best params:", iso_gs.best_params_)

# ----- 5. Predict & Evaluate -----
print("\n🏆 Evaluating on full dataset (unsupervised)...")
raw_pred = model.predict(X)
# map IsolationForest output: +1 → normal (0), -1 → anomaly (1)
y_pred = pd.Series(raw_pred).map({1: 0, -1: 1}).astype(int)

print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1 Score :", f1_score(y, y_pred))

# ----- 6. Visualize Anomalies -----
print("📊 Plotting anomalies on latitude/longitude...")
plt.figure(figsize=(8,6))
plt.scatter(
    v['longitude'], v['latitude'],
    c=y_pred, cmap='coolwarm', s=20, alpha=0.7
)
plt.title("Phase 1: Vessel Anomaly Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()

plot_path = 'dashboard/vessel_anomalies_phase1.png'
plt.savefig(plot_path)
plt.show()
print(f"✅ Plot saved to {plot_path}")

# ----- 7. Save Model -----
model_path = 'models/vessel_anomaly_model_phase1.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
