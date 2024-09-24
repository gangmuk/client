package main

import (
	"context"
	"fmt"
	"os"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

func main() {
	// Check if both SHA256 hash and URI are provided as arguments
	if len(os.Args) < 3 {
		fmt.Println("SHA256 hash and URI not provided. Usage: go run main.go <sha256> <uri>")
		return
	}
	sha256Hash := os.Args[1]
	wasmUri := os.Args[2]

	// Load the kubeconfig file (for local or remote Kubernetes cluster)
	home := homedir.HomeDir()
	kubeconfig := fmt.Sprintf("%s/.kube/config", home)
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		fmt.Printf("Failed to load kubeconfig: %v\n", err)
		return
	}

	// Create the Kubernetes client
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		fmt.Printf("Failed to create Kubernetes client: %v\n", err)
		return
	}

	// Define variables
	configMapName := "shared-span-bootstrap-config"
	namespace := "default"

	// Step 1: Get the existing ConfigMap
	configMap, err := clientset.CoreV1().ConfigMaps(namespace).Get(context.TODO(), configMapName, metav1.GetOptions{})
	if err != nil {
		fmt.Printf("Failed to get ConfigMap: %v\n", err)
		return
	}

	// Step 2: Retrieve the current custom_bootstrap.json content
	bootstrapConfig, ok := configMap.Data["custom_bootstrap.json"]
	if !ok {
		fmt.Println("custom_bootstrap.json not found in ConfigMap")
		return
	}

	// Step 3: Use gjson to verify that the sha256 and uri fields exist
	sha256JsonPath := "bootstrap_extensions.0.typed_config.config.vm_config.code.remote.sha256"
	currentSha256 := gjson.Get(bootstrapConfig, sha256JsonPath)
	if !currentSha256.Exists() {
		fmt.Println("sha256 field not found in the JSON")
		return
	}

	uriJsonPath := "bootstrap_extensions.0.typed_config.config.vm_config.code.remote.http_uri.uri"
	currentUri := gjson.Get(bootstrapConfig, uriJsonPath)
	if !currentUri.Exists() {
		fmt.Println("URI field not found in the JSON")
		return
	}

	// Step 4: Use sjson to set the new sha256 and URI values in the JSON
	updatedBootstrapConfig, err := sjson.Set(bootstrapConfig, sha256JsonPath, sha256Hash)
	if err != nil {
		fmt.Printf("Failed to set new sha256 value in JSON: %v\n", err)
		return
	}

	updatedBootstrapConfig, err = sjson.Set(updatedBootstrapConfig, uriJsonPath, wasmUri)
	if err != nil {
		fmt.Printf("Failed to set new URI value in JSON: %v\n", err)
		return
	}

	// Step 5: Update the ConfigMap with the new JSON configuration
	configMap.Data["custom_bootstrap.json"] = updatedBootstrapConfig

	// Step 6: Update the ConfigMap in Kubernetes
	_, err = clientset.CoreV1().ConfigMaps(namespace).Update(context.TODO(), configMap, metav1.UpdateOptions{})
	if err != nil {
		fmt.Printf("Failed to update ConfigMap: %v\n", err)
		return
	}

	fmt.Println("ConfigMap updated successfully with new Wasm SHA256 hash and URI.")
}
