def predict_fault(model, scaler, sensor_values):

    # Scale sensor data
    scaled_data = scaler.transform([sensor_values])

    # Predict
    prediction = model.predict(scaled_data)

    labels = {
        0: "Healthy Motor",
        1: "Faulty Motor"
    }

    return labels.get(prediction[0], "Unknown")
