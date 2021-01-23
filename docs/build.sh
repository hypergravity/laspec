sphinx-apidoc -o source ../laspec ../laspec/old ../laspec/extern ../laspec/neural_network.py ../laspec/nn.py ../laspec/slam_model.py  -f
make clean html
open ./build/html/index.html