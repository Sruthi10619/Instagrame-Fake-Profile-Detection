import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import joblib
from .models import UserProfile  # Import your UserProfile model

def feature_engineering(df):
    # Assuming 'user_id' is present in your dataset
    user_profiles = UserProfile.objects.all().values()

    # Create a DataFrame from the UserProfile data
    user_profiles_df = pd.DataFrame.from_records(user_profiles)

    # Merge the existing DataFrame with the UserProfile data based on user_id
    df = pd.merge(df, user_profiles_df, left_on='user_id', right_on='user_id', how='left')

    # Perform feature engineering
    df['nums/length username'] = df['instagram_username'].apply(lambda x: len([c for c in x if c.isnumeric()]) / max(len(x), 1))
    df['fullname words'] = df['full_name'].apply(lambda x: len(str(x).split()))
    df['nums/length fullname'] = df['full_name'].apply(lambda x: len([c for c in x if c.isnumeric()]) / max(len(x), 1))
    df['name==username'] = (df['instagram_username'] == df['full_name']).astype(int)
    df['description length'] = df['description'].apply(lambda x: len(str(x)))
    df['external URL'] = df['external_url'].apply(lambda x: 1 if pd.notnull(x) else 0)
    df['private'] = df['private'].astype(int)
    
    # Add other feature engineering steps based on your requirements

    return df


def train_fake_profile_model(df):
    # Feature engineering
    df = feature_engineering(df)

    # Prepare data
    X = df.drop(columns=['fake'])
    y = df['fake']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # Build and compile the model
    model = Sequential()
    model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))  # Adjust input_dim based on the number of features
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, verbose=1, validation_split=0.1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')

    return model, scaler

def load_model(model_path):
    """
    Function to load the trained model from the specified path.
    """
    try:
        model = joblib.load('D:\fake_detection\fake_profile_detection\trained_model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_scaler(scaler_path):
    """
    Function to load the scaler from the specified path.
    """
    try:
        scaler = joblib.load('D:\fake_detection\fake_profile_detection\trained_scaler.pkl')
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        return None

def fetch_user_details(instagram_username):
    app_id = "231431096705774"
    app_secret = "2cd5250c2420be90e54c642f16936251"
    access_token = "IGQWRORHhQajNyVTExX2piTjQ4ZAWRrN1ZAoa1FTY0hPMlRybWwxRm00eUdHQ01IRHowVEw0WklQMnVUcVE5X3JJczQ2TTdDend5Y1VUNFNodTd4V2xDYXF4VjRpZAHp0c1BoblF5WGZAzeGx0N1dNd2N3NzdCWEt0eDgZD"

    url = f"https://graph.instagram.com/{instagram_username}?fields=id,username,name,profile_picture_url,biography,website,follows_count,followed_by_count,media_count&access_token={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        user_data = response.json()
        return user_data
    else:
        print(f"Error fetching user details for '{instagram_username}': {response.text}")
        return None

def predict_fake_profile(instagram_username):
    user_data = fetch_user_details(instagram_username)
    if user_data:
        # Extract relevant information from the user data
        profile_picture_url = user_data.get('profile_picture_url')
        description = user_data.get('biography', '')
        external_url = user_data.get('website', '')
        private = False  # You may need to adjust this based on the Instagram API response
        posts_count = user_data.get('media_count', 0)
        followers_count = user_data.get('followed_by_count', 0)
        follows_count = user_data.get('follows_count', 0)
        
        # Perform fake profile detection using the retrieved information
        prediction = predict_fake_profile_with_details(instagram_username, profile_picture_url, description, external_url, private, posts_count, followers_count, follows_count)
        
        return prediction
    else:
        return 'Error: User not found'



def predict_fake_profile_with_details(instagram_username, profile_picture_url, description, external_url, private, posts_count, followers_count, follows_count):
    # Load the trained model
    model_path = 'D:\fake_detection\fake_profile_detection\trained_model.pkl'  # Replace with the actual path to your trained model
    scaler_path = 'D:\fake_detection\fake_profile_detection\trained_scaler.pkl'  # Replace with the actual path to your scaler

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    if model is None or scaler is None:
        return 'Error: Model or scaler loading failed'

    # Preprocess the input data
    user_data = {
        'instagram_username': [instagram_username],
        'profile_picture_url': [profile_picture_url],
        'description': [description],
        'external_url': [external_url],
        'private': [private],
        'posts_count': [posts_count],
        'followers_count': [followers_count],
        'follows_count': [follows_count],
    }
    df = pd.DataFrame(user_data)

    # Perform feature engineering
    df = feature_engineering(df)

    # Standardize features using the scaler
    X_new = scaler.transform(df.drop(columns=['fake']))

    # Make predictions
    prediction = model.predict(X_new)

    # Interpret the prediction
    if prediction[0][1] > prediction[0][0]:
        return 'Fake'
    else:
        return 'Not Fake'
