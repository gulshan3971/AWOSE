
# Install Python3 and pip3
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Install virtualenv to create an isolated environment
pip3 install virtualenv

# Create a virtual environment for Py-EVM
virtualenv pyevm-env
source pyevm-env/bin/activate

# Install Py-EVM
pip install py-evm

# Initialize and run the Py-EVM node
python3 -m evm.main

echo "Py-EVM setup complete. The node is up and running."
