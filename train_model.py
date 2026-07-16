from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report # EVALUATE PERFORMANCE
import joblib

# Load data
X, y = load_and_preprocess("paysim_small.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) # STRATIFY TO MAINTAIN CLASS PROPORTIONS KEEPS SAME FRAUD 

# Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1) #USE ALL CPUS FOR FASTER TRAINING

# Hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300],# NUMBER OF TREES 
    'max_depth': [10, 20, None], # TREE DEPTH
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized Search CV
rs = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3, # 3-FOLD CROSS-VALIDATION
    scoring='f1', #FOCUS ON FRAUD DETECTION PERFORMANCE
    random_state=42,
    n_jobs=-1
)

# Train
rs.fit(X_train, y_train)

# Best model
best_model = rs.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("Best Parameters:", rs.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl")
