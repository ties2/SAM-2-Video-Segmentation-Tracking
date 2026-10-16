import io
import time

import joblib
import requests
from sklearn.linear_model import LinearRegression

from src.serving.dependency import get_artifact_storage

BASE_URL = "http://localhost:8000"
model_name = f"live-demo-{int(time.time())}"
version = "v1.0.0"

print(f"1. Creating and saving actual model [{model_name}]...")
clf = LinearRegression()
clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
buf = io.BytesIO()
joblib.dump(clf, buf)

storage = get_artifact_storage()
file_name = f"{model_name}_{version}.joblib"
storage.save(f"local://{file_name}", buf.getvalue())

print("2. Registering model...")
requests.post(f"{BASE_URL}/models", json={"name": model_name, "description": "Grafana Data"})

print("3. Registering version...")
requests.post(
    f"{BASE_URL}/models/{model_name}/versions",
    json={
        "version": version,
        "framework": "scikit-learn",
        "model_format": "joblib",
        "artifact_uri": f"local://{file_name}",
    },
)

print("4. Promoting to Production...")
res = requests.post(f"{BASE_URL}/models/{model_name}/versions/{version}/promote")
print(f"   -> Promotion Result: {res.status_code} - {res.json()}")

if res.status_code == 200:
    print("5. Generating 100 Predictions (simulating users)...")
    for _ in range(100):
        requests.post(f"{BASE_URL}/models/{model_name}/predict", json={"features": [10.0, 5.0]})
        time.sleep(0.05)
    print("Traffic generated successfully! 🚀 Wait 15s and check Grafana.")
else:
    print("Promotion failed! Check the error above.")
