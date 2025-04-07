#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <gmp.h>
#include <iomanip>
#include <hip/hip_runtime.h>

// Number of CPU threads to use for operations not offloaded to GPU
const int NUM_THREADS = std::thread::hardware_concurrency();

// Number of Miller-Rabin rounds for 128-bit security
constexpr int MR_ROUNDS = 64;

// GPU Constants
constexpr int GPU_BLOCK_SIZE = 256;
constexpr int MAX_MP_LIMBS = 64;  // Maximum limbs for multi-precision arithmetic (4096 bits)

// Thread synchronization
std::mutex mtx;
std::atomic<bool> is_composite{false};
std::atomic<uint64_t> progress_counter{0};
std::atomic<uint64_t> total_tasks{0};

// Multi-precision integer structure for GPU
struct MP_INT {
    uint32_t limbs[MAX_MP_LIMBS];
    int n_limbs;
};

// HIP device functions for multi-precision arithmetic
__device__ void mp_add(MP_INT* result, const MP_INT* a, const MP_INT* b) {
    uint64_t carry = 0;
    int max_limbs = max(a->n_limbs, b->n_limbs);

    for (int i = 0; i < max_limbs; i++) {
        uint64_t sum = carry;
        if (i < a->n_limbs) sum += a->limbs[i];
        if (i < b->n_limbs) sum += b->limbs[i];

        result->limbs[i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
    }

    if (carry > 0 && max_limbs < MAX_MP_LIMBS) {
        result->limbs[max_limbs] = static_cast<uint32_t>(carry);
        result->n_limbs = max_limbs + 1;
    } else {
        result->n_limbs = max_limbs;
    }
}

__device__ void mp_sub(MP_INT* result, const MP_INT* a, const MP_INT* b) {
    int64_t borrow = 0;

    for (int i = 0; i < a->n_limbs; i++) {
        int64_t diff = static_cast<int64_t>(a->limbs[i]) - (i < b->n_limbs ? b->limbs[i] : 0) - borrow;

        if (diff < 0) {
            diff += (1ULL << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }

        result->limbs[i] = static_cast<uint32_t>(diff);
    }

    // Compute actual size (removing leading zeros)
    result->n_limbs = a->n_limbs;
    while (result->n_limbs > 0 && result->limbs[result->n_limbs - 1] == 0) {
        result->n_limbs--;
    }
    if (result->n_limbs == 0) {
        result->n_limbs = 1;  // Ensure at least one limb
    }
}

__device__ void mp_mul(MP_INT* result, const MP_INT* a, const MP_INT* b) {
    // Clear result
    for (int i = 0; i < a->n_limbs + b->n_limbs; i++) {
        result->limbs[i] = 0;
    }

    // Schoolbook multiplication algorithm
    for (int i = 0; i < a->n_limbs; i++) {
        uint32_t carry = 0;

        for (int j = 0; j < b->n_limbs && (i + j) < MAX_MP_LIMBS; j++) {
            uint64_t product = static_cast<uint64_t>(a->limbs[i]) * b->limbs[j] +
            result->limbs[i + j] + carry;

            result->limbs[i + j] = static_cast<uint32_t>(product);
            carry = product >> 32;
        }

        if (carry > 0 && (i + b->n_limbs) < MAX_MP_LIMBS) {
            result->limbs[i + b->n_limbs] = carry;
        }
    }

    result->n_limbs = min(a->n_limbs + b->n_limbs, MAX_MP_LIMBS);
    while (result->n_limbs > 0 && result->limbs[result->n_limbs - 1] == 0) {
        result->n_limbs--;
    }
    if (result->n_limbs == 0) {
        result->n_limbs = 1;  // Ensure at least one limb
    }
}

__device__ int mp_cmp(const MP_INT* a, const MP_INT* b) {
    if (a->n_limbs > b->n_limbs) return 1;
    if (a->n_limbs < b->n_limbs) return -1;

    for (int i = a->n_limbs - 1; i >= 0; i--) {
        if (a->limbs[i] > b->limbs[i]) return 1;
        if (a->limbs[i] < b->limbs[i]) return -1;
    }

    return 0;  // Equal
}

__device__ void mp_div_rem(MP_INT* quotient, MP_INT* remainder,
                           const MP_INT* dividend, const MP_INT* divisor) {
    // Initialize remainder = dividend
    for (int i = 0; i < dividend->n_limbs; i++) {
        remainder->limbs[i] = dividend->limbs[i];
    }
    remainder->n_limbs = dividend->n_limbs;

    // Initialize quotient = 0
    quotient->limbs[0] = 0;
    quotient->n_limbs = 1;

    // Cannot divide by zero
    if (divisor->n_limbs == 1 && divisor->limbs[0] == 0) {
        return;
    }

    // If dividend < divisor, quotient = 0, remainder = dividend
    if (mp_cmp(dividend, divisor) < 0) {
        return;
    }

    // Binary long division algorithm
    MP_INT temp;
    MP_INT current_divisor;

    for (int i = 0; i < MAX_MP_LIMBS; i++) {
        quotient->limbs[i] = 0;
        current_divisor.limbs[i] = divisor->limbs[i];
    }
    current_divisor.n_limbs = divisor->n_limbs;

    int shift = dividend->n_limbs - divisor->n_limbs;

    // Left shift divisor to align with dividend
    for (int i = current_divisor.n_limbs - 1; i >= 0; i--) {
        current_divisor.limbs[i + shift] = current_divisor.limbs[i];
    }
    for (int i = 0; i < shift; i++) {
        current_divisor.limbs[i] = 0;
    }
    current_divisor.n_limbs += shift;

    quotient->n_limbs = shift + 1;

    while (shift >= 0) {
        if (mp_cmp(remainder, &current_divisor) >= 0) {
            mp_sub(remainder, remainder, &current_divisor);
            quotient->limbs[shift] |= 1;
        }

        // Right shift current_divisor
        for (int i = 0; i < current_divisor.n_limbs - 1; i++) {
            current_divisor.limbs[i] = current_divisor.limbs[i + 1];
        }
        current_divisor.n_limbs--;
        if (current_divisor.n_limbs == 0) {
            current_divisor.n_limbs = 1;
            current_divisor.limbs[0] = 0;
        }

        shift--;
    }

    // Normalize quotient size
    while (quotient->n_limbs > 1 && quotient->limbs[quotient->n_limbs - 1] == 0) {
        quotient->n_limbs--;
    }
                           }

                           __device__ void mp_mod(MP_INT* result, const MP_INT* a, const MP_INT* mod) {
                               MP_INT quotient;
                               mp_div_rem(&quotient, result, a, mod);
                           }

                           __device__ void mp_powm(MP_INT* result, const MP_INT* base, const MP_INT* exp, const MP_INT* mod) {
                               // Initialize result to 1
                               result->limbs[0] = 1;
                               for (int i = 1; i < MAX_MP_LIMBS; i++) {
                                   result->limbs[i] = 0;
                               }
                               result->n_limbs = 1;

                               // If exponent is 0, return 1
                               if (exp->n_limbs == 1 && exp->limbs[0] == 0) {
                                   return;
                               }

                               // Copy base to avoid modifying the original
                               MP_INT base_copy;
                               for (int i = 0; i < base->n_limbs; i++) {
                                   base_copy.limbs[i] = base->limbs[i];
                               }
                               base_copy.n_limbs = base->n_limbs;

                               // Ensure base < mod
                               mp_mod(&base_copy, &base_copy, mod);

                               // Square-and-multiply algorithm
                               for (int i = exp->n_limbs - 1; i >= 0; i--) {
                                   uint32_t exp_limb = exp->limbs[i];

                                   for (int bit = 31; bit >= 0; bit--) {
                                       // Square
                                       MP_INT temp;
                                       mp_mul(&temp, result, result);
                                       mp_mod(result, &temp, mod);

                                       // Multiply if bit is set
                                       if ((exp_limb >> bit) & 1) {
                                           mp_mul(&temp, result, &base_copy);
                                           mp_mod(result, &temp, mod);
                                       }
                                   }
                               }
                           }

                           // Convert GMP mpz_t to MP_INT for GPU
                           void mpz_to_mp_int(mpz_t src, MP_INT* dst) {
                               // Get the number of limbs needed
                               size_t count = mpz_size(src);
                               dst->n_limbs = count * 2;  // GMP uses 64-bit limbs, we use 32-bit

                               // Extract limbs from mpz_t
                               mp_limb_t* limbs = mpz_limbs_read(src);

                               for (size_t i = 0; i < count && i * 2 < MAX_MP_LIMBS; i++) {
                                   dst->limbs[i * 2] = limbs[i] & 0xFFFFFFFF;
                                   dst->limbs[i * 2 + 1] = limbs[i] >> 32;
                               }
                           }

                           // HIP kernel for Miller-Rabin primality test
                           __global__ void miller_rabin_kernel(MP_INT* n, MP_INT* n_minus_1, MP_INT* d,
                                                               uint64_t s, MP_INT* bases,
                                                               bool* results, int num_bases) {
                               int idx = blockIdx.x * blockDim.x + threadIdx.x;
                               if (idx >= num_bases) return;

                               // Copy base to thread-local memory
                               MP_INT a;
                               for (int i = 0; i < bases[idx].n_limbs; i++) {
                                   a.limbs[i] = bases[idx].limbs[i];
                               }
                               a.n_limbs = bases[idx].n_limbs;

                               // Compute a^d mod n
                               MP_INT y;
                               mp_powm(&y, &a, d, n);

                               // If a^d ≡ 1 (mod n) or a^d ≡ -1 (mod n), n is probably prime
                               if ((y.n_limbs == 1 && y.limbs[0] == 1) || mp_cmp(&y, n_minus_1) == 0) {
                                   results[idx] = true;
                                   return;
                               }

                               // Check a^(2^r * d) for r = 1 to s-1
                               for (uint64_t r = 1; r < s; r++) {
                                   // y = y^2 mod n
                                   MP_INT temp;
                                   mp_mul(&temp, &y, &y);
                                   mp_mod(&y, &temp, n);

                                   // If y ≡ -1 (mod n), n is probably prime
                                   if (mp_cmp(&y, n_minus_1) == 0) {
                                       results[idx] = true;
                                       return;
                                   }

                                   // If y ≡ 1 (mod n), n is composite
                                   if (y.n_limbs == 1 && y.limbs[0] == 1) {
                                       results[idx] = false;
                                       return;
                                   }
                               }

                               // If we've reached this point, n is definitely composite
                               results[idx] = false;
                                                               }

                                                               // Class for primality testing of extremely large numbers
                                                               class PrimalityTester {
                                                               private:
                                                                   gmp_randstate_t rng;
                                                                   mpz_t n;           // The number to test
                                                                   mpz_t n_minus_1;   // n-1
                                                                   mpz_t a;           // Base for Miller-Rabin
                                                                   mpz_t y;           // Temporary for calculations
                                                                   mpz_t r;           // For division
                                                                   mpz_t j;           // For exponentiation
                                                                   mpz_t two;         // Constant 2

                                                                   // GPU resources
                                                                   MP_INT* d_n = nullptr;
                                                                   MP_INT* d_n_minus_1 = nullptr;
                                                                   MP_INT* d_d = nullptr;
                                                                   MP_INT* d_bases = nullptr;
                                                                   bool* d_results = nullptr;

                                                                   // Helper function to find factor s where n-1 = 2^s * d
                                                                   uint64_t find_s_d(mpz_t d) {
                                                                       uint64_t s = 0;
                                                                       mpz_set(d, n_minus_1);

                                                                       while (mpz_even_p(d)) {
                                                                           mpz_divexact_ui(d, d, 2);
                                                                           s++;
                                                                       }

                                                                       return s;
                                                                   }

                                                                   // Initialize GPU resources
                                                                   void init_gpu_resources() {
                                                                       // Allocate GPU memory for multi-precision numbers
                                                                       hipMalloc(&d_n, sizeof(MP_INT));
                                                                       hipMalloc(&d_n_minus_1, sizeof(MP_INT));
                                                                       hipMalloc(&d_d, sizeof(MP_INT));
                                                                       hipMalloc(&d_bases, MR_ROUNDS * sizeof(MP_INT));
                                                                       hipMalloc(&d_results, MR_ROUNDS * sizeof(bool));
                                                                   }

                                                                   // Free GPU resources
                                                                   void free_gpu_resources() {
                                                                       if (d_n) hipFree(d_n);
                                                                       if (d_n_minus_1) hipFree(d_n_minus_1);
                                                                       if (d_d) hipFree(d_d);
                                                                       if (d_bases) hipFree(d_bases);
                                                                       if (d_results) hipFree(d_results);

                                                                       d_n = nullptr;
                                                                       d_n_minus_1 = nullptr;
                                                                       d_d = nullptr;
                                                                       d_bases = nullptr;
                                                                       d_results = nullptr;
                                                                   }

                                                               public:
                                                                   PrimalityTester() {
                                                                       // Initialize GMP variables
                                                                       mpz_init(n);
                                                                       mpz_init(n_minus_1);
                                                                       mpz_init(a);
                                                                       mpz_init(y);
                                                                       mpz_init(r);
                                                                       mpz_init(j);
                                                                       mpz_init_set_ui(two, 2);

                                                                       // Initialize random state
                                                                       gmp_randinit_mt(rng);

                                                                       // Seed the random number generator
                                                                       std::random_device rd;
                                                                       gmp_randseed_ui(rng, rd());

                                                                       // Initialize GPU resources
                                                                       init_gpu_resources();
                                                                   }

                                                                   ~PrimalityTester() {
                                                                       // Free GMP variables
                                                                       mpz_clear(n);
                                                                       mpz_clear(n_minus_1);
                                                                       mpz_clear(a);
                                                                       mpz_clear(y);
                                                                       mpz_clear(r);
                                                                       mpz_clear(j);
                                                                       mpz_clear(two);

                                                                       // Clear random state
                                                                       gmp_randclear(rng);

                                                                       // Free GPU resources
                                                                       free_gpu_resources();
                                                                   }

                                                                   // Set the number to test
                                                                   void set_number(const std::string& num_str) {
                                                                       mpz_set_str(n, num_str.c_str(), 10);
                                                                       mpz_sub_ui(n_minus_1, n, 1);
                                                                   }

                                                                   // Perform primality check using GPU-accelerated Miller-Rabin tests
                                                                   bool is_prime(int rounds = MR_ROUNDS) {
                                                                       // Check for small primes
                                                                       if (mpz_cmp_ui(n, 2) == 0 || mpz_cmp_ui(n, 3) == 0) {
                                                                           return true;
                                                                       }

                                                                       // Check if n is even or less than 2
                                                                       if (mpz_even_p(n) || mpz_cmp_ui(n, 2) < 0) {
                                                                           return false;
                                                                       }

                                                                       // Special case for n = 1
                                                                       if (mpz_cmp_ui(n, 1) == 0) {
                                                                           return false;
                                                                       }

                                                                       // First, try division by small primes for quick rejection
                                                                       std::vector<uint64_t> small_primes = {
                                                                           3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
                                                                       };

                                                                       for (uint64_t p : small_primes) {
                                                                           mpz_mod_ui(r, n, p);
                                                                           if (mpz_cmp_ui(r, 0) == 0 && mpz_cmp_ui(n, p) != 0) {
                                                                               return false;
                                                                           }
                                                                       }

                                                                       // Write n-1 as 2^s * d where d is odd
                                                                       mpz_t d;
                                                                       mpz_init(d);
                                                                       uint64_t s = find_s_d(d);

                                                                       // Prepare data for GPU
                                                                       MP_INT h_n, h_n_minus_1, h_d;
                                                                       mpz_to_mp_int(n, &h_n);
                                                                       mpz_to_mp_int(n_minus_1, &h_n_minus_1);
                                                                       mpz_to_mp_int(d, &h_d);

                                                                       // Copy data to GPU
                                                                       hipMemcpy(d_n, &h_n, sizeof(MP_INT), hipMemcpyHostToDevice);
                                                                       hipMemcpy(d_n_minus_1, &h_n_minus_1, sizeof(MP_INT), hipMemcpyHostToDevice);
                                                                       hipMemcpy(d_d, &h_d, sizeof(MP_INT), hipMemcpyHostToDevice);

                                                                       // Generate random bases for Miller-Rabin test
                                                                       std::vector<MP_INT> h_bases(rounds);
                                                                       for (int i = 0; i < rounds; i++) {
                                                                           // Generate random base a in [2, n-2]
                                                                           do {
                                                                               mpz_urandomm(a, rng, n_minus_1);
                                                                           } while (mpz_cmp_ui(a, 1) <= 0);

                                                                           // Convert to MP_INT
                                                                           mpz_to_mp_int(a, &h_bases[i]);
                                                                       }

                                                                       // Copy bases to GPU
                                                                       hipMemcpy(d_bases, h_bases.data(), rounds * sizeof(MP_INT), hipMemcpyHostToDevice);

                                                                       // Reset test results
                                                                       std::vector<bool> h_results(rounds, true);
                                                                       hipMemcpy(d_results, h_results.data(), rounds * sizeof(bool), hipMemcpyHostToDevice);

                                                                       // Launch GPU kernel for Miller-Rabin tests
                                                                       int blocks = (rounds + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                                                                       miller_rabin_kernel<<<blocks, GPU_BLOCK_SIZE>>>(
                                                                           d_n, d_n_minus_1, d_d, s, d_bases, d_results, rounds
                                                                       );

                                                                       // Copy results back from GPU
                                                                       hipMemcpy(h_results.data(), d_results, rounds * sizeof(bool), hipMemcpyDeviceToHost);

                                                                       // Check if any test failed (indicating composite)
                                                                       bool is_probable_prime = true;
                                                                       for (int i = 0; i < rounds; i++) {
                                                                           if (!h_results[i]) {
                                                                               is_probable_prime = false;
                                                                               break;
                                                                           }
                                                                       }

                                                                       // Clean up
                                                                       mpz_clear(d);

                                                                       return is_probable_prime;
                                                                   }

                                                                   // Get the number as a string
                                                                   std::string get_number_str() const {
                                                                       char* str = mpz_get_str(nullptr, 10, n);
                                                                       std::string result(str);
                                                                       free(str);
                                                                       return result;
                                                                   }

                                                                   // Get number of digits
                                                                   uint64_t get_num_digits() const {
                                                                       return mpz_sizeinbase(n, 10);
                                                                   }
                                                               };

                                                               // Main function to read numbers from a file and test them
                                                               int main(int argc, char* argv[]) {
                                                                   if (argc < 2) {
                                                                       std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
                                                                       std::cout << "Input file should contain one number per line to test for primality." << std::endl;
                                                                       return 1;
                                                                   }

                                                                   // Open the input file
                                                                   std::string filename = argv[1];
                                                                   std::ifstream infile(filename);

                                                                   if (!infile.is_open()) {
                                                                       std::cerr << "Error: Could not open file " << filename << std::endl;
                                                                       return 1;
                                                                   }

                                                                   PrimalityTester tester;
                                                                   std::string line;
                                                                   int line_count = 0;

                                                                   // Initialize HIP
                                                                   hipInit(0);

                                                                   // Test each number in the file
                                                                   while (std::getline(infile, line)) {
                                                                       line_count++;

                                                                       // Skip empty lines or lines starting with # (comments)
                                                                       if (line.empty() || line[0] == '#') {
                                                                           continue;
                                                                       }

                                                                       // Remove any spaces or commas
                                                                       line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
                                                                       line.erase(std::remove(line.begin(), line.end(), ','), line.end());

                                                                       try {
                                                                           tester.set_number(line);

                                                                           std::cout << "Testing number " << line_count << ": ";

                                                                           uint64_t digits = tester.get_num_digits();
                                                                           std::cout << digits << " digits" << std::endl;

                                                                           if (digits <= 100) {
                                                                               std::cout << "Number: " << tester.get_number_str() << std::endl;
                                                                           } else {
                                                                               std::string num_str = tester.get_number_str();
                                                                               std::cout << "First 50 digits: " << num_str.substr(0, 50) << "..." << std::endl;
                                                                               std::cout << "Last 50 digits: ..." << num_str.substr(num_str.size() - 50) << std::endl;
                                                                           }

                                                                           auto start_time = std::chrono::high_resolution_clock::now();
                                                                           bool is_prime = tester.is_prime();
                                                                           auto end_time = std::chrono::high_resolution_clock::now();

                                                                           auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                                                                           std::cout << "Result: " << (is_prime ? "PROBABLY PRIME" : "COMPOSITE") << std::endl;
                                                                           std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;
                                                                           std::cout << "---------------------------------------------------" << std::endl;
                                                                       } catch (const std::exception& e) {
                                                                           std::cerr << "Error processing line " << line_count << ": " << e.what() << std::endl;
                                                                       }
                                                                   }

                                                                   infile.close();
                                                                   return 0;
                                                               }
