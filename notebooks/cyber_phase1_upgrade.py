import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score

# 1️⃣ Load Dataset
print("📥 Loading cyber intrusion dataset...")
df = pd.read_csv('data/cyber_data.csv')

# 2️⃣ Clean & Inspect Columns
df.columns = df.columns.str.strip().str.lower()
print("✅ Columns present:", list(df.columns))

required = {'ip_address','port','packet_size','intrusion_attempt'}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing columns {missing}, available: {list(df.columns)}")

# 3️⃣ Feature Engineering
print("🚀 Feature engineering: extracting IP octets + selecting numeric features…")
# split IP into four numeric octets
octets = df['ip_address'].str.split('.', expand=True).astype(int)
octets.columns = [f'ip_oct{i+1}' for i in range(4)]

# build feature matrix
X = pd.concat([
    octets[['ip_oct1','ip_oct2']],      # major network segments
    df[['port','packet_size']]
], axis=1)

y = df['intrusion_attempt']  # 0 = no, 1 = intrusion

print("✅ Feature shape:", X.shape)
print("✅ Feature preview:\n", X.head())

# 4️⃣ Train/Test Split
print("🔀 Splitting into train & test sets…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5️⃣ Hyperparameter Tuning
print("⚙️  Running GridSearchCV for RandomForestClassifier…")
param_grid = {
    'n_estimators': [50,100],
    'max_depth': [None, 10],
    'min_samples_split': [2,5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("✅ Best parameters:", grid.best_params_)

# 6️⃣ Evaluation
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:,1]

print("\n🏆 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ ROC AUC:", roc_auc_score(y_test, y_proba))
print("✅ F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, cmap='Blues'
)
plt.title("Cyber Intrusion Confusion Matrix")
plt.tight_layout()
cm_path = 'dashboard/cyber_confusion_matrix_phase1.png'
plt.savefig(cm_path)
plt.show()
print(f"✅ Confusion matrix saved to {cm_path}")

# 7️⃣ Save Model & Predictions
model_path = 'models/cyber_intrusion_model_phase1.pkl'
joblib.dump(best, model_path)
print(f"✅ Model saved to {model_path}")

preds = pd.DataFrame({
    'actual': y_test.reset_index(drop=True),
    'predicted': y_pred
})
preds_path = 'dashboard/cyber_intrusion_predictions_phase1.csv'
preds.to_csv(preds_path, index=False)
print(f"✅ Predictions saved to {preds_path}")
