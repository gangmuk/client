from locust import HttpUser, task, between
from locust.exception import RescheduleTask

class CheckoutUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Adjust wait time between requests if necessary

    @task
    def checkout_cart(self):
        headers = {
            "x-slate-destination": "west"
        }
        path = "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222"
        
        try:
            with self.client.post(path, headers=headers, timeout=2, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Request failed: {e}")
            RescheduleTask()
