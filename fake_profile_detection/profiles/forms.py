# profiles/forms.py
from django import forms

class DetectionForm(forms.Form):
    instagram_username = forms.CharField(label='Enter Instagram Username', max_length=100)
