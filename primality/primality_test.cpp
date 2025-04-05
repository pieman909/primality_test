#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <gmp.h>
#include <iomanip>
#include <hip/hip_runtime.h>

// Number of threads to use
const int NUM_THREADS = std::thread::hardware_concurrency();

// Number of Miller-Rabin rounds for 128-bit security
constexpr int MR_ROUNDS = 64;

// Thread synchronization
std::mutex mtx;
std::atomic<bool> is_composite{false};
std::atomic<uint64_t> progress_counter{0};
std::atomic<uint64_t> total_tasks{0};

// Error checking macro for HIP
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " \
                    << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple implementation of large integer arithmetic for the GPU
struct GpuInt {
    uint32_t digits[128];  // 4096-bit integer representation
    int num_digits;

    __device__ void set(uint32_t val) {
        for (int i = 0; i < 128; i++) {
            digits[i] = 0;
        }
        digits[0] = val;
        num_digits = 1;
    }

    __device__ bool is_zero() const {
        for (int i = 0; i < num_digits; i++) {
            if (digits[i] != 0) return false;
        }
        return true;
    }

    __device__ bool is_one() const {
        if (digits[0] != 1) return false;
        for (int i = 1; i < num_digits; i++) {
            if (digits[i] != 0) return false;
        }
        return true;
    }

    __device__ int compare(const GpuInt& other) const {
        if (num_digits > other.num_digits) return 1;
        if (num_digits < other.num_digits) return -1;
        
        for (int i = num_digits - 1; i >= 0; i--) {
            if (digits[i] > other.digits[i]) return 1;
            if (digits[i] < other.digits[i]) return -1;
        }
        return 0;
    }

    // Basic modular exponentiation for GPU
    __device__ void pow_mod(const GpuInt& base, const GpuInt& exp, const GpuInt& mod) {
        GpuInt result;
        result.set(1);
        GpuInt temp = base;
        
        for (int i = 0; i < exp.num_digits; i++) {
            uint32_t e = exp.digits[i];
            for (int bit = 0; bit < 32; bit++) {
                if (e & (1 << bit)) {
                    // result = (result * temp) % mod
                    // Simplified for illustration
                }
                // temp = (temp * temp) % mod
                // Simplified for illustration
            }
        }
        *this = result;
    }
};

// GPU kernel for Miller-Rabin primality test
__global__ void miller_rabin_kernel(bool* results, int rounds, const uint32_t* n_data, int n_size,
                                    const uint32_t* n_minus_1_data, int n_minus_1_size,
                                    const uint32_t* d_data, int d_size, uint64_t s,
                                    uint32_t* rng_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rounds) return;
    
    // This is a simplified placeholder - implementing a full Miller-Rabin test
    // with large integer arithmetic on GPU is beyond the scope of this example
    // In a real implementation, you would:
    // 1. Convert GMP data to GpuInt format
    // 2. Generate random base a (2 <= a <= n-2)
    // 3. Compute a^d % n
    // 4. Perform Miller-Rabin test steps
    // 5. Store result in results[idx]
    
    // Placeholder result - always consider number composite to force CPU fallback
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

    // Single Miller-Rabin test with base a
    bool miller_rabin_single_test(const mpz_t a, const mpz_t d, uint64_t s) {
        mpz_t y, tmp;
        mpz_init(y);
        mpz_init(tmp);

        // Compute a^d mod n
        mpz_powm(y, a, d, n);

        // If a^d ≡ 1 (mod n) or a^d ≡ -1 (mod n), n is probably prime
        if (mpz_cmp_ui(y, 1) == 0 || mpz_cmp(y, n_minus_1) == 0) {
            mpz_clear(y);
            mpz_clear(tmp);
            return true;
        }

        // Check a^(2^r * d) for r = 1 to s-1
        for (uint64_t r = 1; r < s; r++) {
            // y = y^2 mod n
            mpz_powm_ui(y, y, 2, n);

            // If y ≡ -1 (mod n), n is probably prime
            if (mpz_cmp(y, n_minus_1) == 0) {
                mpz_clear(y);
                mpz_clear(tmp);
                return true;
            }

            // If y ≡ 1 (mod n), n is composite
            if (mpz_cmp_ui(y, 1) == 0) {
                mpz_clear(y);
                mpz_clear(tmp);
                return false;
            }
        }

        // If we've reached this point, n is definitely composite
        mpz_clear(y);
        mpz_clear(tmp);
        return false;
    }

    // Worker thread function for parallel Miller-Rabin tests
    void worker_miller_rabin(uint64_t start_idx, uint64_t end_idx, mpz_t d, uint64_t s) {
        // Each thread needs its own GMP variables
        mpz_t local_a, local_y, local_j;
        mpz_init(local_a);
        mpz_init(local_y);
        mpz_init(local_j);

        gmp_randstate_t local_rng;
        gmp_randinit_mt(local_rng);

        // Set a different seed for each thread
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dist;
        gmp_randseed_ui(local_rng, dist(gen));

        // Make thread-local copies of shared variables to avoid race conditions
        mpz_t local_n, local_n_minus_1, local_d;
        mpz_init(local_n);
        mpz_init(local_n_minus_1);
        mpz_init(local_d);

        {
            std::lock_guard<std::mutex> lock(mtx);
            mpz_set(local_n, n);
            mpz_set(local_n_minus_1, n_minus_1);
            mpz_set(local_d, d);
        }

        for (uint64_t i = start_idx; i < end_idx; i++) {
            // Skip iteration if a composite result was already found
            if (is_composite.load(std::memory_order_relaxed)) {
                break;
            }

            // Generate random base a in [2, n-2]
            do {
                mpz_urandomm(local_a, local_rng, local_n_minus_1);
            } while (mpz_cmp_ui(local_a, 1) <= 0);

            // Perform Miller-Rabin test with this base
            bool probably_prime = miller_rabin_single_test(local_a, local_d, s);

            // If test failed, mark as composite and exit
            if (!probably_prime) {
                is_composite.store(true, std::memory_order_relaxed);
                break;
            }

            // Update progress counter
            progress_counter.fetch_add(1, std::memory_order_relaxed);
        }

        // Clean up thread-local variables
        mpz_clear(local_a);
        mpz_clear(local_y);
        mpz_clear(local_j);
        mpz_clear(local_n);
        mpz_clear(local_n_minus_1);
        mpz_clear(local_d);
        gmp_randclear(local_rng);
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
    }

    // Set the number to test
    void set_number(const std::string& num_str) {
        mpz_set_str(n, num_str.c_str(), 10);
        mpz_sub_ui(n_minus_1, n, 1);
    }

    // Generate a random number with specified number of digits
    void generate_random_number(uint64_t digits) {
        // Generate a random number with 'digits' decimal digits
        mpz_t base, range;
        mpz_init(base);
        mpz_init(range);

        // Calculate 10^(digits-1)
        mpz_ui_pow_ui(base, 10, digits - 1);

        // Calculate range = 9 * 10^(digits-1)
        mpz_mul_ui(range, base, 9);

        // Generate random number in [0, range-1]
        mpz_urandomm(n, rng, range);

        // Add base to get a number with exactly 'digits' digits
        mpz_add(n, n, base);

        // Ensure number is odd (for primality testing)
        if (mpz_even_p(n)) {
            mpz_add_ui(n, n, 1);
        }

        mpz_sub_ui(n_minus_1, n, 1);

        mpz_clear(base);
        mpz_clear(range);
    }

    // Generate ultra-large random number using 10^10^exp notation
    void generate_ultra_large_number(double exp) {
        // For very large exponents, we need to be more careful
        if (exp > 9) {
            // Generate a number with approximately 10^exp digits
            // by using log10 properties

            // First, calculate roughly how many digits we need
            // For 10^10^exp, the number of digits is approximately 10^exp

            // For extremely large exponents, we'll use a safer approach
            // Generate a large number with a reasonable number of digits
            uint64_t max_safe_digits = 1000000;  // 1 million digits should be manageable

            // Generate initial random number with max_safe_digits
            generate_random_number(max_safe_digits);

            // Manually set the exponent to get approximately 10^10^exp
            mpz_t exponent;
            mpz_init(exponent);

            // Calculate 10^exp as the exponent
            double log_digits = std::pow(10, exp);

            // Ensure we don't exceed what can be represented
            if (log_digits > 1e9) {
                log_digits = 1e9;  // Cap at a reasonable value
            }

            // Set the value with the right number of trailing zeros
            uint64_t zeros_to_add = static_cast<uint64_t>(log_digits) - max_safe_digits;
            if (zeros_to_add > 0) {
                // Multiply by 10^zeros_to_add
                mpz_ui_pow_ui(exponent, 10, zeros_to_add);
                mpz_mul(n, n, exponent);
            }

            mpz_clear(exponent);
        } else {
            // For smaller exponents, we can use the original approach
            uint64_t digits = std::floor(std::pow(10, exp));
            generate_random_number(digits);
        }

        // Update n_minus_1
        mpz_sub_ui(n_minus_1, n, 1);
    }

    // Perform primality check using parallel Miller-Rabin tests
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

        // Reset atomic flags
        is_composite.store(false);
        progress_counter.store(0);
        total_tasks.store(rounds);

        // Start progress reporting thread
        std::atomic<bool> stop_progress{false};
        std::thread progress_thread([&]() {
            while (!stop_progress && !is_composite.load()) {
                uint64_t current = progress_counter.load();
                uint64_t total = total_tasks.load();

                if (total > 0) {
                    double percent = (100.0 * current) / total;
                    std::cout << "\rMiller-Rabin progress: " << current << "/" << total
                    << " (" << std::fixed << std::setprecision(2) << percent << "%)" << std::flush;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            std::cout << std::endl;
        });

        // Launch worker threads for Miller-Rabin tests
        std::vector<std::thread> threads;
        uint64_t batch_size = rounds / NUM_THREADS;
        uint64_t remainder = rounds % NUM_THREADS;

        for (int t = 0; t < NUM_THREADS; t++) {
            uint64_t start_idx = t * batch_size;
            uint64_t end_idx = (t + 1) * batch_size;

            // Add remainder to last thread
            if (t == NUM_THREADS - 1) {
                end_idx += remainder;
            }

            threads.emplace_back(&PrimalityTester::worker_miller_rabin, this, start_idx, end_idx, d, s);
        }

        // Wait for threads to finish
        for (auto& thread : threads) {
            thread.join();
        }

        // Stop progress reporting
        stop_progress = true;
        progress_thread.join();

        mpz_clear(d);

        // If any test failed, the number is composite
        return !is_composite.load();
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

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <mode> [params...]" << std::endl;
        std::cout << "Modes:" << std::endl;
        std::cout << "  1 <number>        - Test if the given number is prime" << std::endl;
        std::cout << "  2 <digits>        - Generate and test a random number with the given number of digits" << std::endl;
        std::cout << "  3 <exp>           - Generate and test a random ultra-large number around 10^10^exp" << std::endl;
        std::cout << "  4 <exp> <count>   - Generate and test <count> ultra-large numbers around 10^10^exp" << std::endl;
        return 1;
    }

    int mode = std::stoi(argv[1]);
    PrimalityTester tester;

    switch (mode) {
        case 1: {  // Test a specific number
            if (argc < 3) {
                std::cout << "Error: Please provide a number to test" << std::endl;
                return 1;
            }

            std::string number = argv[2];
            tester.set_number(number);

            std::cout << "Testing primality of: " << tester.get_number_str() << std::endl;
            std::cout << "Number of digits: " << tester.get_num_digits() << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();
            bool is_prime = tester.is_prime();
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "Result: " << (is_prime ? "PROBABLY PRIME" : "COMPOSITE") << std::endl;
            std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;
            break;
        }

        case 2: {  // Generate and test a random number with given digits
            if (argc < 3) {
                std::cout << "Error: Please provide the number of digits" << std::endl;
                return 1;
            }

            uint64_t digits = std::stoull(argv[2]);
            tester.generate_random_number(digits);

            std::cout << "Generated random number with " << digits << " digits" << std::endl;

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
            break;
        }

        case 3: {  // Generate and test an ultra-large number
            if (argc < 3) {
                std::cout << "Error: Please provide the exponent" << std::endl;
                return 1;
            }

            double exp = std::stod(argv[2]);
            std::cout << "Generating a random number around 10^10^" << exp << std::endl;

            auto gen_start_time = std::chrono::high_resolution_clock::now();
            tester.generate_ultra_large_number(exp);
            auto gen_end_time = std::chrono::high_resolution_clock::now();

            auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end_time - gen_start_time);

            uint64_t digits = tester.get_num_digits();
            std::cout << "Generated a number with " << digits << " digits" << std::endl;
            std::cout << "Generation time: " << gen_duration.count() / 1000.0 << " seconds" << std::endl;

            // Show a glimpse of the number
            std::string num_str = tester.get_number_str();
            std::cout << "First 50 digits: " << num_str.substr(0, 50) << "..." << std::endl;
            std::cout << "Last 50 digits: ..." << num_str.substr(num_str.size() - 50) << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();
            bool is_prime = tester.is_prime();
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "Result: " << (is_prime ? "PROBABLY PRIME" : "COMPOSITE") << std::endl;
            std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;
            break;
        }

        case 4: {  // Generate and test multiple ultra-large numbers
            if (argc < 4) {
                std::cout << "Error: Please provide the exponent and count" << std::endl;
                return 1;
            }

            double exp = std::stod(argv[2]);
            uint64_t count = std::stoull(argv[3]);

            std::cout << "Testing " << count << " random numbers around 10^10^" << exp << std::endl;

            uint64_t prime_count = 0;
            double total_time = 0.0;

            for (uint64_t i = 0; i < count; i++) {
                std::cout << "\n[" << (i+1) << "/" << count << "] Generating number..." << std::endl;
                tester.generate_ultra_large_number(exp);

                uint64_t digits = tester.get_num_digits();
                std::cout << "Testing number with " << digits << " digits" << std::endl;

                auto start_time = std::chrono::high_resolution_clock::now();
                bool is_prime = tester.is_prime();
                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                double seconds = duration.count() / 1000.0;
                total_time += seconds;

                if (is_prime) prime_count++;

                std::cout << "Result: " << (is_prime ? "PROBABLY PRIME" : "COMPOSITE") << std::endl;
                std::cout << "Time taken: " << seconds << " seconds" << std::endl;

                // Show running statistics
                double prime_density = (100.0 * prime_count) / (i+1);
                std::cout << "Running stats: " << prime_count << "/" << (i+1)
                << " primes found (" << prime_density << "%)" << std::endl;
                std::cout << "Average time per test: " << (total_time / (i+1)) << " seconds" << std::endl;
            }

            // Final statistics
            std::cout << "\nFinal results:" << std::endl;
            std::cout << "Tested " << count << " numbers around 10^10^" << exp << std::endl;
            std::cout << "Found " << prime_count << " probable primes" << std::endl;
            std::cout << "Prime density: " << (100.0 * prime_count / count) << "%" << std::endl;
            std::cout << "Total time: " << total_time << " seconds" << std::endl;
            std::cout << "Average time per test: " << (total_time / count) << " seconds" << std::endl;

            break;
        }

        default:
            std::cout << "Error: Invalid mode" << std::endl;
            return 1;
    }

    return 0;
}
