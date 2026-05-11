from motorfaultsvm import train_from_csv
from motorfaultsvm import predict_fault

# Train model
model, scaler, accuracy = train_from_csv(
    "../datasets/im_data.csv"
)

print("Model Accuracy:", accuracy)

# Example sensor values
sample = [45, 0.08, 3.1, 1420]

# Predict motor condition
result = predict_fault(
    model,
    scaler,
    sample
)

print("Prediction:", result)
