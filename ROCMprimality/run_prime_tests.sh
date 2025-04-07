#!/bin/bash

# Run Prime Estimation and Testing Pipeline
# This script compiles the prime estimator with HIP/ROCm support,
# compiles the primality tester, runs them in sequence, and logs results

# Parameters
EXPONENT=${1:-9}           # Default: 10^10^9
NUM_SAMPLES=${2:-10000}    # Default: 10,000 samples
NUM_CANDIDATES=${3:-100}   # Default: 100 candidates
INPUT_FILE="numbers.txt"
RESULTS_FILE="prime_results.txt"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}====== PRIME NUMBER ESTIMATION AND TESTING ======${NC}"
echo "Exponent: 10^10^$EXPONENT"
echo "Samples: $NUM_SAMPLES"
echo "Candidates: $NUM_CANDIDATES"

# Check if we have HIP/ROCm
if hipconfig --version &>/dev/null; then
    echo -e "${GREEN}HIP/ROCm detected - compiling with GPU support${NC}"
    HIP_AVAILABLE=true
else
    echo -e "${YELLOW}HIP/ROCm not detected - using CPU-only version${NC}"
    HIP_AVAILABLE=false
fi

# Step 1: Compile prime_estimator.cpp
echo -e "\n${YELLOW}[STEP 1/4] Compiling prime estimator...${NC}"
if [ "$HIP_AVAILABLE" = true ]; then
    hipcc -o prime_estimator prime_estimator.cpp -lgmp -lmpfr -std=c++14 -DUSE_HIP
    COMPILE_RESULT=$?
else
    g++ -o prime_estimator prime_estimator.cpp -lgmp -lmpfr -std=c++14 -pthread
    COMPILE_RESULT=$?
fi

if [ $COMPILE_RESULT -ne 0 ]; then
    echo -e "${RED}Failed to compile prime estimator. Exiting.${NC}"
    exit 1
fi
echo -e "${GREEN}Prime estimator compiled successfully.${NC}"

# Step 2: Compile primality_test.cpp
echo -e "\n${YELLOW}[STEP 2/4] Compiling primality tester...${NC}"
hipcc -o primality_test primality_test.cpp -lgmp -std=c++14
COMPILE_RESULT=$?

if [ $COMPILE_RESULT -ne 0 ]; then
    echo -e "${RED}Failed to compile primality tester. Exiting.${NC}"
    exit 1
fi
echo -e "${GREEN}Primality tester compiled successfully.${NC}"

# Step 3: Run prime estimator
echo -e "\n${YELLOW}[STEP 3/4] Running prime estimator...${NC}"
./prime_estimator $EXPONENT $NUM_SAMPLES $NUM_CANDIDATES $INPUT_FILE
ESTIMATOR_RESULT=$?

if [ $ESTIMATOR_RESULT -ne 0 ]; then
    echo -e "${RED}Prime estimator failed with code $ESTIMATOR_RESULT. Exiting.${NC}"
    exit 1
fi

# Check if numbers file exists and has content
if [ ! -s "$INPUT_FILE" ]; then
    echo -e "${RED}No prime candidates were generated in $INPUT_FILE. Exiting.${NC}"
    exit 1
fi

echo -e "${GREEN}Prime estimator completed. Generated $(wc -l < $INPUT_FILE) candidate numbers.${NC}"

# Step 4: Run primality test
echo -e "\n${YELLOW}[STEP 4/4] Running primality tests...${NC}"
echo "Results will be written to $RESULTS_FILE"

# Initialize results file
echo "# Primality Test Results" > $RESULTS_FILE
echo "# Generated on $(date)" >> $RESULTS_FILE
echo "# Testing numbers around 10^10^$EXPONENT" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Run the tests
./primality_test $INPUT_FILE | tee -a $RESULTS_FILE

# Check for prime numbers
PRIME_COUNT=$(grep -c "PRIME" $RESULTS_FILE)
TOTAL_COUNT=$(wc -l < $INPUT_FILE)

echo -e "\n${YELLOW}====== TEST RESULTS SUMMARY ======${NC}"
echo "Tested $TOTAL_COUNT candidate numbers"
echo "Found $PRIME_COUNT prime numbers"
echo "Success rate: $(echo "scale=2; $PRIME_COUNT * 100 / $TOTAL_COUNT" | bc)%"
echo -e "Detailed results saved to ${GREEN}$RESULTS_FILE${NC}"

echo -e "\n${GREEN}Prime estimation and testing pipeline completed successfully!${NC}"
