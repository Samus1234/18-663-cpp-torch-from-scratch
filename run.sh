echo "Processing Dataset...";

python3 processData.py;

echo "Compiling Code...";

g++ -O3 -march=native -std=c++11 -mavx2 -mfma -flto -fopenmp -fPIC -fno-math-errno -funroll-loops -I/usr/local/include/ main.cpp -o main;

echo "Compile Done, running code...";

./main