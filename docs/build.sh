sphinx-apidoc -o source ../laspec ../laspec/old ../laspec/extern -f
make clean html
open ./build/html/index.html