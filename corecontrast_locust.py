from locust import HttpUser, task, between, LoadTestShape, constant_pacing
import math
import json
import os

addHdrs = {
    "X-Slate-1mb-Writes": "4",
}

class SingleCoreWest(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def singlecore(self):
        self.client.post(
            "/singlecore",
            headers={"x-slate-destination": "west", **addHdrs},
            timeout=2
        )
    
class SingleCoreCentral(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def singlecore(self):
        self.client.post(
            "/singlecore",
            headers={"x-slate-destination": "central", **addHdrs},
            timeout=2
        )

class SingleCoreSouth(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def singlecore(self):
        self.client.post(
            "/singlecore",
            headers={"x-slate-destination": "south", **addHdrs},
            timeout=2
        )

class SingleCoreEast(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def singlecore(self):
        self.client.post(
            "/singlecore",
            headers={"x-slate-destination": "east", **addHdrs},
            timeout=2
        )

class MultiCoreWest(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def multicore(self):
        self.client.post(
            "/multicore",
            headers={"x-slate-destination": "west", **addHdrs},
            timeout=2
        )
    
class MultiCoreCentral(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def multicore(self):
        self.client.post(
            "/multicore",
            headers={"x-slate-destination": "central", **addHdrs},
            timeout=2
        )

class MultiCoreSouth(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def multicore(self):
        self.client.post(
            "/multicore",
            headers={"x-slate-destination": "south", **addHdrs},
            timeout=2
        )

class MultiCoreEast(HttpUser):
    wait_time = constant_pacing(1)

    @task
    def multicore(self):
        self.client.post(
            "/multicore",
            headers={"x-slate-destination": "east", **addHdrs},
            timeout=2
        )

def process_workloads_to_stages(workloads, total_duration):
    """
    Transforms the workloads dict into a list of stages.
    Each stage is a tuple:
    (stage_duration, west_singlecore_rps, west_multicore_rps,
     east_singlecore_rps, east_multicore_rps,
     central_singlecore_rps, central_multicore_rps,
     south_singlecore_rps, south_multicore_rps)
    
    Parameters:
    - workloads: Dict containing per-region, per-request-type RPS configurations.
    - total_duration: Total duration of the experiment in seconds.
    
    Returns:
    - final_stages: List of 9-tuples representing each stage.
    """
    # Collect all unique start times
    unique_times = set()
    for region, req_types in workloads.items():
        for req_type, timings in req_types.items():
            for timing in timings:
                unique_times.add(timing[0])

    sorted_times = sorted(unique_times)

    # Ensure that the first start time is 0
    if sorted_times[0] != 0:
        sorted_times = [0] + sorted_times

    # Initialize current RPS for all regions and request types
    regions = ["west", "east", "central", "south"]
    request_types = ["singlecore", "multicore"]
    current_rps = {region: {req_type: 0 for req_type in request_types} for region in regions}

    stages = []

    # Iterate through each start time and update RPS accordingly
    for time in sorted_times:
        for region, req_types in workloads.items():
            for req_type, timings in req_types.items():
                for timing in timings:
                    if timing[0] == time:
                        current_rps[region][req_type] = timing[1]

        # Create a snapshot of current RPS
        snapshot = {
            "west_singlecore": current_rps["west"]["singlecore"],
            "west_multicore": current_rps["west"]["multicore"],
            "east_singlecore": current_rps["east"]["singlecore"],
            "east_multicore": current_rps["east"]["multicore"],
            "central_singlecore": current_rps["central"]["singlecore"],
            "central_multicore": current_rps["central"]["multicore"],
            "south_singlecore": current_rps["south"]["singlecore"],
            "south_multicore": current_rps["south"]["multicore"],
        }

        stages.append((time, snapshot))

    # Sort stages by start_time
    stages = sorted(stages, key=lambda x: x[0])

    # Merge stages with the same configuration
    merged_stages = []
    previous_snapshot = None
    for stage in stages:
        if previous_snapshot and stage[1] == previous_snapshot:
            continue  # Skip if the configuration hasn't changed
        merged_stages.append(stage)
        previous_snapshot = stage[1]

    # Convert to list of 9-tuples with duration
    final_stages = []
    for i in range(len(merged_stages)):
        stage_time = merged_stages[i][0]
        # Determine the duration of the stage
        if i + 1 < len(merged_stages):
            next_stage_time = merged_stages[i + 1][0]
            duration = next_stage_time - stage_time
        else:
            duration = total_duration - stage_time  # Last stage duration

        # Ensure that duration is positive
        if duration <= 0:
            raise ValueError(f"Invalid stage duration at stage {i}: duration={duration}")

        # Create the 9-tuple (duration, 8 rps)
        stage = (
            duration,
            merged_stages[i][1]["west_singlecore"],
            merged_stages[i][1]["west_multicore"],
            merged_stages[i][1]["east_singlecore"],
            merged_stages[i][1]["east_multicore"],
            merged_stages[i][1]["central_singlecore"],
            merged_stages[i][1]["central_multicore"],
            merged_stages[i][1]["south_singlecore"],
            merged_stages[i][1]["south_multicore"],
        )
        final_stages.append(stage)

    # Validate total duration
    calculated_total = sum(stage[0] for stage in final_stages)
    if calculated_total > total_duration:
        raise ValueError(f"Calculated total duration {calculated_total} exceeds the specified total_duration {total_duration}.")
    elif calculated_total < total_duration:
        # Add a final stage to fill the remaining duration with the last configuration
        last_stage = final_stages[-1]
        remaining_duration = total_duration - calculated_total
        final_stages.append((
            remaining_duration,
            last_stage[1],
            last_stage[2],
            last_stage[3],
            last_stage[4],
            last_stage[5],
            last_stage[6],
            last_stage[7],
            last_stage[8],
        ))

    return final_stages


stages_file = os.getenv("STAGES_FILE", "stages.json")
experiment_duration = os.getenv("EXPERIMENT_DURATION", 60*5)  # 4 hours
with open(stages_file, "r") as f:
    workloads = json.load(f)
stages = process_workloads_to_stages(workloads, experiment_duration)
print("Processed Stages:")
for stage in stages:
    print(stage)


class SmoothTransitionShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.stages = stages
        self.current_stage = 0
        self.stage_start_time = 0

    def tick(self):
        run_time = self.get_run_time()

        if self.current_stage >= len(self.stages):
            return None  # End test when stages are complete

        stage_duration, west_singlecore_rps, west_multicore_rps, \
        east_singlecore_rps, east_multicore_rps, \
        central_singlecore_rps, central_multicore_rps, \
        south_singlecore_rps, south_multicore_rps = self.stages[self.current_stage]

        # Check if the current stage duration has elapsed
        if run_time >= self.stage_start_time + stage_duration:
            # Move to the next stage
            self.stage_start_time += stage_duration
            self.current_stage += 1
            if self.current_stage >= len(self.stages):
                return None  # End test
            stage_duration, west_singlecore_rps, west_multicore_rps, \
            east_singlecore_rps, east_multicore_rps, \
            central_singlecore_rps, central_multicore_rps, \
            south_singlecore_rps, south_multicore_rps = self.stages[self.current_stage]

        # Update weights based on RPS
        total_rps = (
            west_singlecore_rps + west_multicore_rps +
            east_singlecore_rps + east_multicore_rps +
            central_singlecore_rps + central_multicore_rps +
            south_singlecore_rps + south_multicore_rps
        )

        # Prevent division by zero
        if total_rps > 0:
            SingleCoreWest.weight = west_singlecore_rps / total_rps
            MultiCoreWest.weight = west_multicore_rps / total_rps
            SingleCoreEast.weight = east_singlecore_rps / total_rps
            MultiCoreEast.weight = east_multicore_rps / total_rps
            SingleCoreCentral.weight = central_singlecore_rps / total_rps
            MultiCoreCentral.weight = central_multicore_rps / total_rps
            SingleCoreSouth.weight = south_singlecore_rps / total_rps
            MultiCoreSouth.weight = south_multicore_rps / total_rps
            
            
        c = [SingleCoreWest, MultiCoreWest, SingleCoreEast, MultiCoreEast, SingleCoreCentral, MultiCoreCentral, SingleCoreSouth, MultiCoreSouth]
        for i in c:
            print(f"{i.__name__}: {i.weight}")
        return (total_rps, total_rps, c)