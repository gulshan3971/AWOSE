#!/bin/bash

# This script sets up Hyperledger Fabric on a Linux system

# Install dependencies
sudo apt-get update
sudo apt-get install -y curl docker docker-compose

# Download the Hyperledger Fabric binaries and Docker images
curl -sSL https://bit.ly/2ysbOFE | bash -s

# Start the Hyperledger Fabric network
cd fabric-samples/test-network
./network.sh up createChannel -c mychannel -ca

# To deploy a chaincode
./network.sh deployCC -ccn basic -ccp ../asset-transfer-basic/chaincode-go -ccl go

echo "Hyperledger Fabric setup complete. The network is up and running."
