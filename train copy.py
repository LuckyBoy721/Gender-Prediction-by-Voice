import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import json  # Tambahkan pustaka json

# Load dataset
data = np.load("gender_dataset_balanced.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Dictionary to store models and their names
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} accuracy: {accuracy:.2f}")
    
    # Save the model
    model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'.")

# Display comparison of model performance
print("\nModel Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.2f}")

# Save comparison results to a text file
with open("model_comparison.txt", "w") as f:
    f.write("Model Comparison:\n")
    for model_name, accuracy in results.items():
        f.write(f"{model_name}: {accuracy:.2f}\n")
    print("Comparison results saved as 'model_comparison.txt'.")

# Save comparison results to a JSON file
with open("model_accuracies.json", "w") as f:
    json.dump(results, f, indent=4)
    print("Comparison results saved as 'model_accuracies.json'.")
