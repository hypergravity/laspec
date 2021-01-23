sphinx-apidoc -o source ../laspec -f
make clean html
open ./build/html/index.html