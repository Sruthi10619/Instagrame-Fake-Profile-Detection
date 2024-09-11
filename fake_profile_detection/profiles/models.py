# profiles/models.py
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    instagram_user_id = models.CharField(max_length=255, unique=True)
    instagram_username = models.CharField(max_length=255)
    full_name = models.CharField(max_length=255)
    profile_picture_url = models.URLField()
    description = models.TextField(null=True, blank=True)
    external_url = models.URLField(null=True, blank=True)
    private = models.BooleanField(default=False)
    posts_count = models.IntegerField(default=0)
    followers_count = models.IntegerField(default=0)
    follows_count = models.IntegerField(default=0)
    fake = models.IntegerField(default=0)  # Add a new field to store fake prediction

    def __str__(self):
        return self.user.username
