from locust import HttpUser, task, between, LoadTestShape
import math

class CheckoutUserWest(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def checkout_cart(self):
        self.client.post(
            "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222",
            headers={"x-slate-destination": "west"},
            timeout=2
        )

class CheckoutUserEast(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def checkout_cart(self):
        self.client.post(
            "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222",
            headers={"x-slate-destination": "east"},
            timeout=2
        )

class CheckoutUserSouth(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def checkout_cart(self):
        self.client.post(
            "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222",
            headers={"x-slate-destination": "south"},
            timeout=2
        )

class CheckoutUserCentral(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def checkout_cart(self):
        self.client.post(
            "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222",
            headers={"x-slate-destination": "central"},
            timeout=2
        )


class SmoothTransitionShape(LoadTestShape):
    stages = [
        # (stage_time, start_users, end_users, west_weight, east_weight, south_weight, central_weight)
        (30, 50, 400, 0.25, 0.25, 0.25, 0.25),
        (30, 100, 800, 0.25, 0.25, 0.25, 0.25),
        (30, 200, 2000, 0.25, 0.25, 0.25, 0.25),
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage_time, start_users, end_users, west_weight, east_weight, south_weight, central_weight in self.stages:
            if run_time < stage_time:
                # Calculate the proportion of time elapsed in the current stage
                proportion = run_time / stage_time

                # Smoothly transition from start_users to end_users based on the elapsed proportion
                target_users = math.ceil(start_users + proportion * (end_users - start_users))

                # Set dynamic weights for each user class
                CheckoutUserWest.weight = int(west_weight * 100)
                CheckoutUserEast.weight = int(east_weight * 100)
                CheckoutUserSouth.weight = int(south_weight * 100)
                CheckoutUserCentral.weight = int(central_weight * 100)
                
                return (target_users, target_users // 10)  # Adjust spawn rate as needed
            run_time -= stage_time

        return None  # End test when stages are complete
