from kubernetes import client, config
import yaml
replMap = {
    "gcr.io/google-samples/microservices-demo/frontend:v0.10.1": "docker.io/adiprerepa/boutique-frontend:latest",
    "gcr.io/google-samples/microservices-demo/checkoutservice:v0.10.1": "docker.io/adiprerepa/boutique-checkout:latest",
    "gcr.io/google-samples/microservices-demo/recommendationservice:v0.10.1": "docker.io/adiprerepa/boutique-recommendation:latest",
}
def scale_deployments(namespace, replica_count, exclude_deployments=[]):
    # Load kubeconfig (assumes you have access to the cluster)
    config.load_kube_config()

    # Create an instance of the API class
    apps_v1 = client.AppsV1Api()

    # Get all deployments in the specified namespace
    deployments = apps_v1.list_namespaced_deployment(namespace)

    for deployment in deployments.items:
        # Skip deployments in the exclude list
        if deployment.metadata.name in exclude_deployments:
            continue
        
        # Define the new number of replicas
        body = {
            "spec": {
                "replicas": replica_count
            }
        }

        # Scale the deployment
        response = apps_v1.patch_namespaced_deployment_scale(
            name=deployment.metadata.name,
            namespace=namespace,
            body=body
        )

        print(f"Scaled deployment {deployment.metadata.name} to {replica_count} replicas")

def update_deployment_images():
    # Create the AppsV1 API client
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()

    # Get all deployments in all namespaces
    deployments = apps_v1.list_deployment_for_all_namespaces().items

    for deployment in deployments:
        namespace = deployment.metadata.namespace
        name = deployment.metadata.name
        containers = deployment.spec.template.spec.containers
        
        # Flag to determine if an update is needed
        update_needed = False

        # Go through all the containers in the deployment
        for container in containers:
            current_image = container.image
            if current_image in replMap:
                new_image = replMap[current_image]
                print(f"Updating image in deployment {name} in namespace {namespace}: {current_image} -> {new_image}")
                container.image = new_image
                update_needed = True

        # If any container image was updated, patch the deployment
        if update_needed:
            # Patch the deployment with the updated image
            try:
                apps_v1.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body=deployment
                )
                print(f"Successfully updated deployment {name} in namespace {namespace}")
            except Exception as e:
                print(f"Error updating deployment {name} in namespace {namespace}: {e}")

def change_load_balancing_policy_to_round_robin():
    # Load kubeconfig (default location is ~/.kube/config)
    config.load_kube_config()

    # Create an instance of the API class for custom resources
    custom_api = client.CustomObjectsApi()

    # Fetch all DestinationRules across all namespaces
    group = 'networking.istio.io'
    version = 'v1beta1'
    plural = 'destinationrules'

    # Fetch DestinationRules across all namespaces
    destinationrules = custom_api.list_cluster_custom_object(group=group, version=version, plural=plural)

    for rule in destinationrules['items']:
        namespace = rule['metadata']['namespace']
        name = rule['metadata']['name']

        # Ensure 'trafficPolicy' exists in the spec, if not, create it
        if 'trafficPolicy' not in rule['spec']:
            rule['spec']['trafficPolicy'] = {}

        # Ensure 'loadBalancer' exists under 'trafficPolicy', if not, create it
        if 'loadBalancer' not in rule['spec']['trafficPolicy']:
            rule['spec']['trafficPolicy']['loadBalancer'] = {}

        # Update load balancing policy to round robin
        rule['spec']['trafficPolicy']['loadBalancer'] = {
            'simple': 'ROUND_ROBIN'
        }

        # Update the DestinationRule with the modified spec
        custom_api.patch_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name,
            body=rule
        )
        print(f"Updated DestinationRule '{name}' in namespace '{namespace}' to use ROUND_ROBIN load balancing.")

def remove_cpu_limits_from_deployments(namespace='default'):
    # Load the kubeconfig file (make sure you have access to the cluster)
    config.load_kube_config()

    # Initialize the API client for interacting with deployments
    v1_apps = client.AppsV1Api()

    # List all deployments in the given namespace
    deployments = v1_apps.list_namespaced_deployment(namespace=namespace)

    for deployment in deployments.items:
        updated = False  # Flag to check if we modified the deployment

        # Iterate over each container in the deployment
        for container in deployment.spec.template.spec.containers:
            # Set the resources field to None regardless of its current value
            if container.resources is not None:
                container.resources = None
                updated = True
                print(f"Setting resources to None for container {container.name} in deployment {deployment.metadata.name}")

        # Update the deployment if we modified the resources
        if updated:
            # Use a patch to update the deployment spec
            body = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container.name,
                                    "resources": None
                                } for container in deployment.spec.template.spec.containers
                            ]
                        }
                    }
                }
            }

            v1_apps.patch_namespaced_deployment(
                name=deployment.metadata.name, 
                namespace=namespace, 
                body=body
            )
            print(f"Updated deployment {deployment.metadata.name}")
# scale_deployments("default", 4, exclude_deployments=["slate-controller"])
def update_image_pull_policy():
    # Load kube config
    config.load_kube_config()

    # Create an instance of the API class for interacting with deployments
    api_instance = client.AppsV1Api()

    # List all deployments in the default namespace
    namespace = 'default'
    deployments = api_instance.list_namespaced_deployment(namespace=namespace)

    for deployment in deployments.items:
        # Skip the slate-controller deployment
        if deployment.metadata.name == 'slate-controller':
            continue

        # Iterate over containers and update the imagePullPolicy
        updated = False
        for container in deployment.spec.template.spec.containers:
            if container.image_pull_policy != 'IfNotPresent':
                container.image_pull_policy = 'IfNotPresent'
                updated = True
        
        # Patch the deployment only if it was updated
        if updated:
            # Create a patch body
            body = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container.name,
                                    "imagePullPolicy": container.image_pull_policy
                                } for container in deployment.spec.template.spec.containers
                            ]
                        }
                    }
                }
            }

            # Apply the patch to the deployment
            api_instance.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=namespace,
                body=body
            )
            print(f"Updated imagePullPolicy to IfNotPresent for deployment: {deployment.metadata.name}")

# update_image_pull_policy()
remove_cpu_limits_from_deployments()