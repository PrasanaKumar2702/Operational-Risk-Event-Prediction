
# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Preprocessing Function
def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)

    # Encode categorical variables
    encoder = LabelEncoder()
    categorical_columns = ['Event_Type', 'Unit', 'Severity', 'Impact']
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_columns = ['Frequency', 'Resolution_Time']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Save preprocessed data
    data.to_csv(output_path, index=False)
    print("Data preprocessing completed. Processed data saved to:", output_path)

# Training Function
def train_model(input_path, output_dir):
    data = pd.read_csv(input_path)

    # Define features and target
    X = data.drop(columns=['Event_ID', 'Risk_Outcome'])
    y = data['Risk_Outcome']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    dump(rf_model, os.path.join(output_dir, "random_forest_model.joblib"))

    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dump(dt_model, os.path.join(output_dir, "decision_tree_model.joblib"))

    print("Models trained and saved to:", output_dir)

# Evaluation Function
def evaluate_model(input_path, model_path, output_dir):
    data = pd.read_csv(input_path)
    X = data.drop(columns=['Event_ID', 'Risk_Outcome'])
    y = data['Risk_Outcome']

    # Load model
    model = load(model_path)

    # Make predictions
    y_pred = model.predict(X)

    # Evaluate model
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # Save evaluation report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print("Evaluation report saved to:", report_path)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    metrics_path = os.path.join(output_dir, "metrics.png")
    plt.savefig(metrics_path)
    print("Confusion matrix saved to:", metrics_path)

# Main Function
if __name__ == "__main__":
    # File paths
    input_file = "data/operational_logs.csv"
    processed_file = "data/processed_data.csv"
    output_dir = "results"

    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Preprocess data
    preprocess_data(input_file, processed_file)

    # Step 2: Train models
    train_model(processed_file, output_dir)

    # Step 3: Evaluate Random Forest model
    rf_model_path = os.path.join(output_dir, "random_forest_model.joblib")
    evaluate_model(processed_file, rf_model_path, output_dir)
