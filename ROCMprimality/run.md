# Prime Number Estimation and Testing Guide

This system helps estimate and test very large prime numbers using a two-step process:
1. Generate potential prime candidates in a specified range
2. Test these candidates for primality

## Quick Start

1. Ensure you have all required files:
   - `prime_estimator.cpp` - Generates potential prime numbers
   - `primality_test.cpp` - Tests numbers for primality (your existing file)
   - `run_prime_tests.sh` - Automation script

2. Make the script executable:
   ```bash
   chmod +x run_prime_tests.sh
   ```

3. Run with default settings (exponent 9, which means numbers around 10^10^9):
   ```bash
   ./run_prime_tests.sh
   ```

## Usage Options

Basic usage:
```bash
./run_prime_tests.sh [exponent] [num_samples] [num_candidates]
```

Parameters:
- `exponent`: The exponent value for 10^10^n (default: 9)
- `num_samples`: Number of samples for estimation (default: 10000)
- `num_candidates`: Number of prime candidates to generate (default: 100)

Examples:
```bash
# Generate and test numbers around 10^10^12 with 50,000 samples and 200 candidates
./run_prime_tests.sh 12 50000 200

# Generate and test smaller numbers (around 10^10^3)
./run_prime_tests.sh 3

# Generate and test very large numbers with more candidates
./run_prime_tests.sh 20 5000 500
```

## Output Files

- `numbers.txt`: Contains the generated prime candidates
- `prime_results.txt`: Contains the primality test results

## GPU Acceleration

If HIP/ROCm is detected on your system, the script will automatically use GPU acceleration. No additional configuration is needed.

## Memory Requirements

For extremely large numbers (exponent > 20), ensure your system has sufficient memory, as GMP/MPFR libraries require memory proportional to the number size.

## Manual Compilation (If Needed)

If you prefer to compile the components manually:

```bash
# Compile prime estimator with HIP/ROCm support
hipcc -o prime_estimator prime_estimator.cpp -lgmp -lmpfr -std=c++14 -DUSE_HIP

# Compile prime estimator without GPU support
g++ -o prime_estimator prime_estimator.cpp -lgmp -lmpfr -std=c++14 -pthread

# Compile primality test
hipcc -o primality_test primality_test.cpp -lgmp -std=c++14

# Run prime estimator
./prime_estimator 9 10000 100 numbers.txt

# Run primality test and save results
./primality_test numbers.txt > prime_results.txt
```
