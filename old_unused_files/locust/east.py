from locust import HttpUser, task, between, LoadTestShape, constant_pacing
from locust.exception import RescheduleTask

class CheckoutUser(HttpUser):
    # wait_time = between(0.1, 0.5)
    wait_time = constant_pacing(1)

    @task
    def checkout_cart(self):
        try:
            with self.client.post(
                "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222",
                headers={"x-slate-destination": "east"},
                timeout=2,
                catch_response=True
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Request failed: {e}")
            raise RescheduleTask()


# Define other user classes for east, south, central regions similarly

class CustomShape(LoadTestShape):
    """
    Dynamically controls RPS by adjusting user levels and spawn rates over time.
    Each tuple in stages defines (duration, user count, spawn rate).
    """

    stages = [
        (30, 100),   # Ramp up to 100 users over 1 minute
        (30, 150),  # Hold 200 users for 2 minutes
        (30, 200),   # Ramp up to 300 users over 1 minute
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage_time, user_count in self.stages:
            if run_time < stage_time:
                return (user_count, user_count)
            run_time -= stage_time
        return None  # End test when stages are complete