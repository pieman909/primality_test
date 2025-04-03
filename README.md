This repo contains the c++ files as well as their associated binaries and compilation instructions:
prime_estimator.cpp
primality_test.cpp
print.cpp

Along with these files there are precompiled binaries for each of them. 
This is a work in progress and merely working skeleton code for a more 
advanced/optimized approach using Vulkan to parallelize the Miller-Rabin 
testing method in primality_test, improving cross-compatibility across both 
AMD and NVIDIA gpus. This project exists for me to gain more experice with
Vulkan to speed up and parallelize computational tasks, with applications
specific to speeding up inference among llms, physics demos, or graphics 
demos. One unique design feature I used for this project was using multithreading 
to help parallelize computations on the CPU.

Thanks for reading and feel free to test this out!
