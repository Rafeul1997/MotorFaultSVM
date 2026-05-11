import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_from_csv(csv_file):

    # Load dataset
    df = pd.read_csv(csv_file)

    # Replace empty values
    df = df.replace(to_replace={'': '0'})

    # Fill missing values
    df = df.fillna(0)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)

    # Features = all columns except last
    X = df.iloc[:, :-1]

    # Labels = last column
    y = df.iloc[:, -1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # SVM model
    model = SVC(kernel='linear', C=1.0)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy
