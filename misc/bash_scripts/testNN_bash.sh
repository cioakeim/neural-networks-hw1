#!/bin/bash

sizes=(6250 12500 18750 25000 31250 37500 43750 50000)
cd ../build

for size in "${sizes[@]}"; do
  echo "Testing with training size: $size"
  ./testNN "$size" 
done
