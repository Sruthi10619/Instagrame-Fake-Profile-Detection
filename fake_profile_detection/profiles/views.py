# profiles/views.py
from .instagram_api import InstagramAPI
from .fake_profile_detection import feature_engineering, train_fake_profile_model
from django.shortcuts import render
import pandas as pd
import numpy as np
from .forms import DetectionForm
import os
import joblib
from .fake_profile_detection import predict_fake_profile
def fetch_and_train(request):
    app_id = "231431096705774"
    app_secret = "2cd5250c2420be90e54c642f16936251"
    access_token = "IGQWRORHhQajNyVTExX2piTjQ4ZAWRrN1ZAoa1FTY0hPMlRybWwxRm00eUdHQ01IRHowVEw0WklQMnVUcVE5X3JJczQ2TTdDend5Y1VUNFNodTd4V2xDYXF4VjRpZAHp0c1BoblF5WGZAzeGx0N1dNd2N3NzdCWEt0eDgZD"

    instagram_api = InstagramAPI(app_id=app_id, app_secret=app_secret, access_token=access_token)
    user_data = instagram_api.get_user_data("user_id")

    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])

    # Feature engineering
    user_df = feature_engineering(user_df)

    # Train the fake profile detection model
    model, scaler = train_fake_profile_model(pd.read_csv('train.csv'))

    # Make predictions on the new user data
    new_user_data = user_df.drop(columns=['fake'])
    new_user_data = scaler.transform(new_user_data)
    new_user_prediction = model.predict(new_user_data)
    predicted_value = np.argmax(new_user_prediction)

    return render(request, 'result_template.html', {'result': 'Fake' if predicted_value == 1 else 'Not Fake'})
def real_time_detection(request):
    if request.method == 'POST':
        form = DetectionForm(request.POST)
        if form.is_valid():
            instagram_username = form.cleaned_data['instagram_username']
            result = predict_fake_profile(instagram_username)
            return render(request, 'result_template.html', {'result': result})
    else:
        form = DetectionForm()
    
    return render(request, 'detection_form.html', {'form': form})
# In your views.py
from .fake_profile_detection import train_fake_profile_model

def train_and_save_model(request):
    # Assuming you have a method to fetch the dataset df
    df = fetch_dataset()  
    
    # Train the model
    model, scaler = train_fake_profile_model(df)
    
    # Define paths to save the trained model and scaler
    model_path = os.path.join('D:\fake_detection\fake_profile_detection', 'trained_model.pkl')
    scaler_path = os.path.join('D:\fake_detection\fake_profile_detection', 'trained_scaler.pkl')

    # Save the trained model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Optionally, you can render a response indicating that the training was successful
    return render(request, 'training_success.html', {'model_path': model_path, 'scaler_path': scaler_path})
def fetch_dataset():
    # Assuming your dataset is stored in a CSV file named 'dataset.csv' in the same directory as your Django app
    dataset_path = 'D:\fake_detection\fake_profile_detection\profiles\train.csv'

    # Load the dataset into a DataFrame
    df = pd.read_csv(dataset_path)

    return df
from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("This is the index page of the profiles app.")
def profile_detail(request, username):
    return HttpResponse(f"This is the profile detail page for user {username}.")
def index(request):
    # Modify this function to render a different template or return different content
    return HttpResponse("This is the homepage. Modify this view function to render a different template or content.")
def index(request):
    # Render the 'homepage.html' template
    return render(request, 'homepage.html')