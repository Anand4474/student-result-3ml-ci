import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Demonstration student dataset
data = pd.DataFrame({
    "study_hours": [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11],
    "attendance": [55,60,65,62,70,68,72,75,78,80,82,84,86,88,90,91,93,95,96,98],
    "result": [
        0,0,0,0,0,0,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1
    ]
})

data.to_csv("student_results.csv", index=False)

X = data[["study_hours", "attendance"]]
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "student_result_model.pkl")

metrics = {
    "accuracy": round(float(accuracy), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model trained successfully.")
print(json.dumps(metrics, indent=4))
