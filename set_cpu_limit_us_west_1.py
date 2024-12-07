from kubernetes import client, config


def remove_cpu_limits_from_deployments(namespace='default'):
    config.load_kube_config()
    v1_apps = client.AppsV1Api()
    deployments = v1_apps.list_namespaced_deployment(namespace=namespace)
    for deployment in deployments.items:
        updated = False  # Flag to check if we modified the deployment
        for container in deployment.spec.template.spec.containers:
            if container.resources is not None:
                container.resources = None
                updated = True
                print(f"Setting resources to None for container {container.name} in deployment {deployment.metadata.name}")
        if updated:
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


def set_cpu_limit(deploy, cpu_limit, cluster):
    config.load_kube_config()
    api_instance = client.AppsV1Api()
    deployments = api_instance.list_deployment_for_all_namespaces()
    for deployment in deployments.items:
        if cluster in deployment.metadata.name and deploy in deployment.metadata.name:
            body = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": deployment.spec.template.spec.containers[0].name,
                                    "resources": {
                                        "limits": {"cpu": cpu_limit}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
            api_instance.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=deployment.metadata.namespace,
                body=body
            )
            print(f"set_cpu_limit: deployment {deployment.metadata.name}")
            
            
# if __name__ == '__main__':
#     set_cpu_limit_us_west_1("301m")