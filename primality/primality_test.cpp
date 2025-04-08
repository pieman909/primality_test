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
struct MP_INT_GPU {
    uint32_t limbs[MAX_MP_LIMBS];
    int n_limbs;
};

// HIP device functions for multi-precision arithmetic
__device__ void mp_add_gpu(MP_INT_GPU* result, const MP_INT_GPU* a, const MP_INT_GPU* b) {
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

__device__ void mp_sub_gpu(MP_INT_GPU* result, const MP_INT_GPU* a, const MP_INT_GPU* b) {
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

__device__ void mp_mul_gpu(MP_INT_GPU* result, const MP_INT_GPU* a, const MP_INT_GPU* b) {
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

__device__ int mp_cmp_gpu(const MP_INT_GPU* a, const MP_INT_GPU* b) {
    if (a->n_limbs > b->n_limbs) return 1;
    if (a->n_limbs < b->n_limbs) return -1;

    for (int i = a->n_limbs - 1; i >= 0; i--) {
        if (a->limbs[i] > b->limbs[i]) return 1;
        if (a->limbs[i] < b->limbs[i]) return -1;
    }

    return 0;  // Equal
}

__device__ void mp_div_rem_gpu(MP_INT_GPU* quotient, MP_INT_GPU* remainder,
                           const MP_INT_GPU* dividend, const MP_INT_GPU* divisor) {
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
    if (mp_cmp_gpu(dividend, divisor) < 0) {
        return;
    }

    // Binary long division algorithm
    MP_INT_GPU temp;
    MP_INT_GPU current_divisor;

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
        if (mp_cmp_gpu(remainder, &current_divisor) >= 0) {
            mp_sub_gpu(remainder, remainder, &current_divisor);
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

__device__ void mp_mod_gpu(MP_INT_GPU* result, const MP_INT_GPU* a, const MP_INT_GPU* mod) {
    MP_INT_GPU quotient;
    mp_div_rem_gpu(&quotient, result, a, mod);
}

__device__ void mp_powm_gpu(MP_INT_GPU* result, const MP_INT_GPU* base, const MP_INT_GPU* exp, const MP_INT_GPU* mod) {
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
    MP_INT_GPU base_copy;
    for (int i = 0; i < base->n_limbs; i++) {
        base_copy.limbs[i] = base->limbs[i];
    }
    base_copy.n_limbs = base->n_limbs;

    // Ensure base < mod
    mp_mod_gpu(&base_copy, &base_copy, mod);

    // Square-and-multiply algorithm
    for (int i = exp->n_limbs - 1; i >= 0; i--) {
        uint32_t exp_limb = exp->limbs[i];

        for (int bit = 31; bit >= 0; bit--) {
            // Square
            MP_INT_GPU temp;
            mp_mul_gpu(&temp, result, result);
            mp_mod_gpu(result, &temp, mod);

            // Multiply if bit is set
            if ((exp_limb >> bit) & 1) {
                MP_INT_GPU temp;
                mp_mul_gpu(&temp, result, &base_copy);
                mp_mod_gpu(result, &temp, mod);
            }
        }
    }
}

// Convert GMP mpz_t to MP_INT_GPU for GPU
void mpz_to_mp_int_gpu(const mpz_t src, MP_INT_GPU* dst) { // Changed to const mpz_t
    // Get the number of limbs needed
    size_t count = mpz_size(src);
    dst->n_limbs = count * 2;  // GMP uses 64-bit limbs, we use 32-bit

    // Extract limbs from mpz_t
    const mp_limb_t* limbs = mpz_limbs_read(src);
    for (size_t i = 0; i < count && i * 2 < MAX_MP_LIMBS; i++) {
        dst->limbs[i * 2] = limbs[i] & 0xFFFFFFFF;
        dst->limbs[i * 2 + 1] = limbs[i] >> 32;
    }
     if (count * 2 > MAX_MP_LIMBS) {
        std::cerr << "Warning: Input number too large for MP_INT_GPU, truncating.\n";
    }
}

// Convert MP_INT_GPU to GMP mpz_t
void mp_int_gpu_to_mpz(const MP_INT_GPU* src, mpz_t dest) {
    // Calculate the number of 64-bit limbs
    size_t limbs_count = (src->n_limbs + 1) / 2;

     // Allocate enough space for the limbs.
    mpz_realloc(dest, limbs_count);

    // Get a pointer to the limbs array of the mpz_t.
    mp_limb_t *dest_limbs = mpz_limbs_write(dest, limbs_count);
    size_t i;
    for (i = 0; i < limbs_count; i++) {
        dest_limbs[i] = (mp_limb_t)(src->limbs[i * 2]) |
            ((mp_limb_t)(src->limbs[i * 2 + 1]) << 32);
    }
    mpz_set_size(dest, i);
    mpz_set_si(dest, 1);
}

// HIP kernel for Miller-Rabin primality test
__global__ void miller_rabin_kernel(MP_INT_GPU* n, MP_INT_GPU* n_minus_1, MP_INT_GPU* d, uint64_t s, MP_INT_GPU* bases, bool* results, int num_bases) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bases) return;

    // Copy base to thread-local memory
    MP_INT_GPU a;
    for (int i = 0; i < bases[idx].n_limbs; i++) {
        a.limbs[i] = bases[idx].limbs[i];
    }
    a.n_limbs = bases[idx].n_limbs;

    // Compute a^d mod n
    MP_INT_GPU y;
    mp_powm_gpu(&y, &a, d, n);

    // If a^d ≡ 1 (mod n) or a^d ≡ -1 (mod n), n is probably prime
    if ((y.n_limbs == 1 && y.limbs[0] == 1) || mp_cmp_gpu(&y, n_minus_1) == 0) {
        results[idx] = true;
        return;
    }

    // Check a^(2^r * d) for r = 1 to s-1
    for (uint64_t r = 1; r < s; r++) {
        // y = y^2 mod n
        MP_INT_GPU temp;
        mp_mul_gpu(&temp, &y, &y);
        mp_mod_gpu(&y, &temp, n);

        // If y ≡ -1 (mod n), n is probably prime
        if (mp_cmp_gpu(&y, n_minus_1) == 0) {
            results[idx] = true;
            return;
        }
    }

    // If none of the above conditions are met, n is composite
    results[idx] = false;
}

// Function to perform Miller-Rabin primality test on a single number
bool miller_rabin(const mpz_t n) {
    if (mpz_cmp_ui(n, 2) < 0) return false; // Numbers less than 2 are not prime
    if (mpz_cmp_ui(n, 2) == 0) return true;  // 2 is prime
    if (mpz_even_p(n)) return false;       // Even numbers greater than 2 are not prime

    mpz_t n_minus_1;
    mpz_init(n_minus_1);
    mpz_sub_ui(n_minus_1, n, 1); // n - 1

    mpz_t d;
    mpz_init(d);
    uint64_t s = 0;

    // Find d and s such that n - 1 = 2^s * d
    mpz_set(d, n_minus_1);
    while (mpz_even_p(d)) {
        mpz_divexact_ui(d, d, 2);
        s++;
    }

    // Choose random bases a
    gmp_randstate_t rand_state;  // Declare random state variable
    gmp_randinit_default(rand_state);  // Initialize random state
    std::vector<mpz_t> bases(MR_ROUNDS);
    for (int i = 0; i < MR_ROUNDS; i++) {
        mpz_init(bases[i]);
        mpz_urandomm(bases[i], rand_state, n); // 1 <= a < n
    }
    gmp_randclear(rand_state); // Clear random state

    // Convert GMP numbers to GPU format
    std::vector<MP_INT_GPU> bases_gpu(MR_ROUNDS);
    MP_INT_GPU n_gpu, n_minus_1_gpu, d_gpu;
    mpz_to_mp_int_gpu(n, &n_gpu);
    mpz_to_mp_int_gpu(n_minus_1, &n_minus_1_gpu);
    mpz_to_mp_int_gpu(d, &d_gpu);
    for (int i = 0; i < MR_ROUNDS; i++) {
        mpz_to_mp_int_gpu(bases[i], &bases_gpu[i]);
    }

    // Allocate memory on GPU for bases and results
    MP_INT_GPU* d_gpu_ptr;
    MP_INT_GPU* n_gpu_ptr;
    MP_INT_GPU* n_minus_1_gpu_ptr;
    MP_INT_GPU* bases_gpu_ptr;
    bool* results_gpu_ptr;

    hipMalloc(&n_gpu_ptr, sizeof(MP_INT_GPU));
    hipMalloc(&n_minus_1_gpu_ptr, sizeof(MP_INT_GPU));
    hipMalloc(&d_gpu_ptr, sizeof(MP_INT_GPU));
    hipMalloc(&bases_gpu_ptr, MR_ROUNDS * sizeof(MP_INT_GPU));
    hipMalloc(&results_gpu_ptr, MR_ROUNDS * sizeof(bool));

    // Copy data to GPU
    hipMemcpy(n_gpu_ptr, &n_gpu, sizeof(MP_INT_GPU), hipMemcpyHostToDevice);
    hipMemcpy(n_minus_1_gpu_ptr, &n_minus_1_gpu, sizeof(MP_INT_GPU), hipMemcpyHostToDevice);
    hipMemcpy(d_gpu_ptr, &d_gpu, sizeof(MP_INT_GPU), hipMemcpyHostToDevice);
    hipMemcpy(bases_gpu_ptr, bases_gpu.data(), MR_ROUNDS * sizeof(MP_INT_GPU), hipMemcpyHostToDevice);

    // Launch the kernel
    int num_blocks = (MR_ROUNDS + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    miller_rabin_kernel<<<num_blocks, GPU_BLOCK_SIZE>>>(
        n_gpu_ptr, n_minus_1_gpu_ptr, d_gpu_ptr, s, bases_gpu_ptr, results_gpu_ptr, MR_ROUNDS);

    // Copy results back from GPU
    std::vector<bool> results(MR_ROUNDS);
    hipMemcpy(results.data(), results_gpu_ptr, MR_ROUNDS * sizeof(bool), hipMemcpyDeviceToHost);

    // Check if any result is false
    bool is_prime = true;
    for (int i = 0; i < MR_ROUNDS; i++) {
        if (!results[i]) {
            is_prime = false;
            break;
        }
    }

    // Free GPU memory
    hipFree(n_gpu_ptr);
    hipFree(n_minus_1_gpu_ptr);
    hipFree(d_gpu_ptr);
    hipFree(bases_gpu_ptr);
    hipFree(results_gpu_ptr);

    // Clear GMP variables
    mpz_clear(n_minus_1);
    mpz_clear(d);
    for (int i = 0; i < MR_ROUNDS; i++) {
        mpz_clear(bases[i]);
    }
    return is_prime;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << input_file << std::endl;
        return 1;
    }

    std::string line;
    int line_count = 0;
    while (std::getline(infile, line)) {
        line_count++;
        try {
            // Convert input string to mpz_t
            mpz_t n;
            mpz_init(n);
            if (mpz_set_str(n, line.c_str(), 10) != 0) {
                throw std::runtime_error("Invalid input: Not a valid number");
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            bool is_prime = miller_rabin(n);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "Result: " << (is_prime ? "PROBABLY PRIME" : "COMPOSITE") << std::endl;
            std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;
            std::cout << "---------------------------------------------------" << std::endl;

            mpz_clear(n); // Clear the mpz_t
        } catch (const std::exception& e) {
            std::cerr << "Error processing line " << line_count << ": " << e.what() << std::endl;
        }
    }

    infile.close();
    return 0;
}
