from src.pipeline.train_pipeline import predict
import numpy as np
from src.components.data_preperation import X_val, y_val,X_train

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_val_np = X_val.to_numpy()
y_val_np = y_val.to_numpy()

X_val_np = scaler.transform(X_val.to_numpy())

probs, blah= predict(X_val_np)

predictions = np.argmax(probs, axis=1)

# 4. Compare with Ground Truth
print(f"First 5 Predictions: {predictions[:5]}")
print(f"First 5 Actuals:     {y_val_np[:5]}")

# 5. Calculate Accuracy
accuracy = np.mean(predictions == y_val_np)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")