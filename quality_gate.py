import json
import sys

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]
threshold = 0.80

print(f"Model accuracy: {accuracy}")
print(f"Required accuracy: {threshold}")

if accuracy < threshold:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
