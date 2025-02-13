import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Define column names for the dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# Load dataset
def load_dataset(filepath):
    print("Loading dataset...")
    try:
        data = pd.read_csv(filepath, header=None, names=columns)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please check the path.")
        exit()

# Preprocess dataset
def preprocess_data(data):
    print("Preprocessing data...")
    
    # Encode categorical features
    categorical_features = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le
    
    # Encode target label for binary classification
    data['label'] = data['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # Separate features and target
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Preprocessing complete.")
    return X_scaled, y

# Main code
if __name__ == "__main__":
    # Specify the dataset path
    dataset_path = r"C:\Users\User\Desktop\IDS new\dataset\kddcup.data_10_percent_corrected"
    
    # Load and preprocess the dataset
    data = load_dataset(dataset_path)
    X, y = preprocess_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Display dataset information
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Save processed data (features and labels)
    output_path = r"C:\Users\User\Desktop\IDS new\processed_data.csv"
    processed_data = pd.DataFrame(X, columns=data.columns[:-1])  # Use original feature names
    processed_data['label'] = y  # Add the label column
    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
