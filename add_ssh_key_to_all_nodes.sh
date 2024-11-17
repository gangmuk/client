#!/bin/bash

# SSH key to add to each node's authorized_keys
SSH_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC21JO2iVQRkmSLRqcIBPcMlQNI1HqcdFcTowNKNYmh0LhX7eSVnLAvNcsdMC+UZcvXYeSPYkOhoxU6/jgWl3H9wo12Z3/fl8Q/GStdR9NGF5coEsTfmxvAplVwJrSFxioiI8z6t4qZV0NU11qYcfUgqPcdQtAElem909w46GvIZ2P02l/biVKxMX8rVgfbJeUvmFStX/HFPDbxQc7nQonJwMya9MHQmvZtH7WqJ7kOjNXBnoVYyduvb9rJoQPx3PjGnSLzO3AFxeZdBQVeJmoOdNmV2p4ahW9MwBiy5LnuSCqhCv0/oJEKX+w8jsq1Y/L8sX25BiBIcNxSxL2whPbN gangmuk@node0.slate1.istio-pg0.clemson.cloudlab.us"

# List of host names for each node
HOSTNAMES=(
  "node1.slate1.istio-pg0.clemson.cloudlab.us"
  "node2.slate1.istio-pg0.clemson.cloudlab.us"
  "node3.slate1.istio-pg0.clemson.cloudlab.us"
  "node4.slate1.istio-pg0.clemson.cloudlab.us"
  "node5.slate1.istio-pg0.clemson.cloudlab.us"
)

# Loop through each host and add the SSH key
for HOST in "${HOSTNAMES[@]}"; do
  echo "Adding SSH key to $HOST"
  ssh "gangmuk@$HOST" "echo '$SSH_KEY' >> ~/.ssh/authorized_keys"
done

echo "SSH key added to all nodes."
