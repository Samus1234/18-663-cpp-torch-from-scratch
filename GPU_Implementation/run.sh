echo "Processing Dataset...";

python3 processData.py;

echo "Compiling Code...";

nvcc -O3 -std=c++11 -I/usr/local/include/ main.cu -o main

echo "Compile Done, running code...";

./main