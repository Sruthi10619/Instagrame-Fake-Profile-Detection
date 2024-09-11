
import requests

class InstagramAPI:
    BASE_URL = "https://graph.instagram.com/v12.0"

    def __init__(self, app_id, app_secret, access_token):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token

    def _make_request(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["access_token"] = self.access_token

        response = requests.get(url, params=params)
        response_data = response.json()

        if response.status_code != 200:
            raise Exception(f"Instagram API Error: {response_data.get('error', 'Unknown error')}")

        return response_data

    def get_user_info(self):
        endpoint = "/me"
        return self._make_request(endpoint)

    def get_user_data(self, user_id):
        endpoint = f"/{user_id}"
        return self._make_request(endpoint)

    def get_user_media(self, user_id):
        endpoint = f"/{user_id}/media"
        return self._make_request(endpoint)

    # Add more methods to interact with the Instagram API as needed


# Example Usage:
# instagram_api = InstagramAPI(app_id="your_app_id", app_secret="your_app_secret", access_token="your_access_token")
# user_info = instagram_api.get_user_info()
# print(user_info)
