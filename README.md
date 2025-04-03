This repo contains the c++ files as well as their associated binaries and compilation instructions:

prime_estimator.cpp 

  Responsible for generating potential prime numbers within a given range
  
primality_test.cpp

  Evaluates a given potential prime number for primality
  
print.cpp

  Internal tool for evaluating single-threaded and multi-threaded performance

==========================================================================================

Along with these files there are precompiled binaries for each of them. 
This is a work in progress and merely working skeleton code for a more 
advanced/optimized approach using Vulkan to parallelize the Miller-Rabin 
testing method in primality_test, improving cross-compatibility across both 
AMD and NVIDIA gpus. This project exists for me to gain more experice with
Vulkan to speed up and parallelize computational tasks, with applications
specific to speeding up inference among llms, physics demos, or graphics 
demos. One unique design feature I used for this project was using multithreading 
to help parallelize computations on the CPU. As of right now, this implementation
is slower for computations than single threading because of computaional overhead,
but after I create speedups in Vulkan, the primality_test program should run
a little faster.

Thanks for reading and feel free to test this out!
