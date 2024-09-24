package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	// "istio.io/client-go/pkg/apis/extensions/v1alpha1"
	"istio.io/client-go/pkg/clientset/versioned"
)

const (
	namespace  = "default"
	pluginName = "slate-wasm-plugin"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <env var> <value>")
		os.Exit(1)
	}

	hillClimbingValue := os.Args[1]

	// Load kubeconfig
	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
	if envKubeconfig := os.Getenv("KUBECONFIG"); envKubeconfig != "" {
		kubeconfig = envKubeconfig
	}

	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		panic(err.Error())
	}

	// Create Istio client
	istioClient, err := versioned.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	// Get the WasmPlugin
	wasmPlugin, err := istioClient.ExtensionsV1alpha1().WasmPlugins(namespace).Get(context.TODO(), pluginName, metav1.GetOptions{})
	if err != nil {
		panic(err.Error())
	}

	// Update HILLCLIMBING environment variable
	updated := false
	for i, envVar := range wasmPlugin.Spec.VmConfig.Env {
		if envVar.Name == "HILLCLIMBING" {
			wasmPlugin.Spec.VmConfig.Env[i].Value = hillClimbingValue
			updated = true
			break
		}
	}

    if !updated {
        fmt.Printf("HILLCLIMBING environment variable not found in %s\n", pluginName)
        return
    }

	// Update the WasmPlugin
	_, err = istioClient.ExtensionsV1alpha1().WasmPlugins(namespace).Update(context.TODO(), wasmPlugin, metav1.UpdateOptions{})
	if err != nil {
		panic(err.Error())
	}

	fmt.Printf("Successfully updated HILLCLIMBING to %s\n", hillClimbingValue)
}
