#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <gmp.h>
#include <mpfr.h>
#include <iomanip>
#include <fstream>
#include <string>

// Optional HIP/ROCm integration
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

// Set precision for MPFR
constexpr int MPFR_PRECISION = 16384;

// Number of threads to use
const int NUM_THREADS = std::thread::hardware_concurrency();

// Thread synchronization
std::mutex mtx;
std::atomic<uint64_t> progress_counter{0};
std::atomic<uint64_t> total_tasks{0};

// Class for statistical prime estimation
class PrimeEstimator {
private:
    // MPFR variables for high-precision arithmetic
    mpfr_t x_lower, x_upper, li_result, log_x, sqrt_x, result;
    mpfr_t tmp1, tmp2, tmp3, correction;

    // Random generators for Monte Carlo sampling
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> dist;

    // Parameters for estimation
    uint64_t num_samples;
    double confidence_level;
    std::string output_file;
    uint64_t num_candidates;

    // Helper function to calculate logarithmic integral li(x)
    void logarithmic_integral(mpfr_t result, mpfr_t x) {
        // Check for zero or negative values
        if (mpfr_cmp_ui(x, 0) <= 0) {
            mpfr_set_ui(result, 0, MPFR_RNDN);
            return;
        }

        mpfr_log(log_x, x, MPFR_RNDN);

        // Check for division by zero
        if (mpfr_cmp_ui(log_x, 0) == 0) {
            mpfr_set_ui(result, 0, MPFR_RNDN);
            return;
        }
        // Use the approximation: li(x) ≈ x/ln(x) * (1 + 1/ln(x) + 2/ln(x)^2 + 6/ln(x)^3 + ...)
        mpfr_log(log_x, x, MPFR_RNDN);

        // Initialize result = x / ln(x)
        mpfr_div(result, x, log_x, MPFR_RNDN);

        // Calculate correction terms
        mpfr_set_ui(tmp1, 1, MPFR_RNDN);
        mpfr_div(tmp1, tmp1, log_x, MPFR_RNDN);  // 1/ln(x)

        mpfr_set_ui(tmp2, 2, MPFR_RNDN);
        mpfr_div(log_x, log_x, log_x, MPFR_RNDN); // ln(x)^2
        mpfr_div(tmp2, tmp2, log_x, MPFR_RNDN);  // 2/ln(x)^2

        mpfr_set_ui(tmp3, 6, MPFR_RNDN);
        mpfr_div(log_x, log_x, log_x, MPFR_RNDN); // ln(x)^3
        mpfr_div(tmp3, tmp3, log_x, MPFR_RNDN);  // 6/ln(x)^3

        // Combine correction terms
        mpfr_add(correction, tmp1, tmp2, MPFR_RNDN);
        mpfr_add(correction, correction, tmp3, MPFR_RNDN);
        mpfr_add_ui(correction, correction, 1, MPFR_RNDN);

        // Apply correction to result
        mpfr_mul(result, result, correction, MPFR_RNDN);
    }

    // Apply Riemann hypothesis-based correction
    void apply_riemann_correction(mpfr_t result, mpfr_t x) {
        // Use approximation based on first few zeros of Riemann zeta function
        mpfr_sqrt(sqrt_x, x, MPFR_RNDN);
        mpfr_log(log_x, x, MPFR_RNDN);

        // Term for first zero: li(x^0.5+14.1i)
        // Simplified approximation for demonstration
        mpfr_div(tmp1, sqrt_x, log_x, MPFR_RNDN);
        mpfr_mul_ui(tmp1, tmp1, 8, MPFR_RNDN);
        mpfr_cos(tmp2, log_x, MPFR_RNDN);
        mpfr_mul(tmp1, tmp1, tmp2, MPFR_RNDN);

        // Subtract correction
        mpfr_sub(result, result, tmp1, MPFR_RNDN);
    }

    // Generate a potential prime number in the specified range
    void generate_potential_prime(mpfr_t result, mpfr_t lower, mpfr_t upper) {
        mpfr_t range, random_offset;
        mpfr_init2(range, MPFR_PRECISION);
        mpfr_init2(random_offset, MPFR_PRECISION);

        // Calculate range
        mpfr_sub(range, upper, lower, MPFR_RNDN);

        // Generate random number in [0,1]
        double rand_val = dist(rng);
        mpfr_set_d(random_offset, rand_val, MPFR_RNDN);

        // Scale to range
        mpfr_mul(random_offset, random_offset, range, MPFR_RNDN);

        // Add to lower bound
        mpfr_add(result, lower, random_offset, MPFR_RNDN);

        // Make it odd (potential prime)
        mpfr_floor(result, result);
        if (mpfr_get_ui(result, MPFR_RNDN) % 2 == 0) {
            mpfr_add_ui(result, result, 1, MPFR_RNDN);
        }

        mpfr_clear(range);
        mpfr_clear(random_offset);
    }

    // Process a single range
    void process_range(uint64_t start_idx, uint64_t end_idx, std::vector<std::pair<double, double>>& results) {
        mpfr_t local_lower, local_upper, count, density;
        mpfr_init2(local_lower, MPFR_PRECISION);
        mpfr_init2(local_upper, MPFR_PRECISION);
        mpfr_init2(count, MPFR_PRECISION);
        mpfr_init2(density, MPFR_PRECISION);

        std::vector<std::pair<double, double>> local_results;

        for (uint64_t i = start_idx; i < end_idx; i++) {
            // Generate sample range for estimation
            double alpha = dist(rng);
            double beta = dist(rng);

            if (alpha > beta) std::swap(alpha, beta);

            // Convert to MPFR
            mpfr_set_d(tmp1, alpha, MPFR_RNDN);
            mpfr_set_d(tmp2, beta, MPFR_RNDN);

            // Calculate actual range bounds
            mpfr_mul(tmp1, tmp1, x_upper, MPFR_RNDN);
            mpfr_add(local_lower, x_lower, tmp1, MPFR_RNDN);

            mpfr_mul(tmp2, tmp2, x_upper, MPFR_RNDN);
            mpfr_add(local_upper, x_lower, tmp2, MPFR_RNDN);

            // Calculate prime count estimation
            logarithmic_integral(tmp1, local_upper);
            logarithmic_integral(tmp2, local_lower);
            mpfr_sub(count, tmp1, tmp2, MPFR_RNDN);

            // Apply Riemann correction
            apply_riemann_correction(count, local_upper);
            apply_riemann_correction(count, local_lower);

            // Calculate prime density
            mpfr_sub(tmp3, local_upper, local_lower, MPFR_RNDN);
            mpfr_div(density, count, tmp3, MPFR_RNDN);

            // Store results
            double est_count = mpfr_get_d(count, MPFR_RNDN);
            double est_density = mpfr_get_d(density, MPFR_RNDN);

            local_results.emplace_back(est_count, est_density);

            // Update progress
            progress_counter.fetch_add(1, std::memory_order_relaxed);
        }

        // Thread-safe update of results
        {
            std::lock_guard<std::mutex> lock(mtx);
            results.insert(results.end(), local_results.begin(), local_results.end());
        }

        mpfr_clear(local_lower);
        mpfr_clear(local_upper);
        mpfr_clear(count);
        mpfr_clear(density);
    }

    // Generate candidate prime numbers
    void generate_candidates() {
        std::ofstream outfile(output_file);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return;
        }

        mpfr_t candidate;
        mpfr_init2(candidate, MPFR_PRECISION);
        char* str_buffer = new char[MPFR_PRECISION/2]; // Buffer for string representation

        std::cout << "Generating " << num_candidates << " prime candidates..." << std::endl;

        for (uint64_t i = 0; i < num_candidates; i++) {
            generate_potential_prime(candidate, x_lower, x_upper);

            // Convert to string and write to file
            mpfr_get_str(str_buffer, nullptr, 10, 0, candidate, MPFR_RNDN);
            outfile << str_buffer << std::endl;

            // Update progress
            if ((i + 1) % 10 == 0 || i == num_candidates - 1) {
                std::cout << "\rGenerated " << (i + 1) << "/" << num_candidates
                << " candidates (" << std::fixed << std::setprecision(1)
                << (100.0 * (i + 1) / num_candidates) << "%)" << std::flush;
            }
        }
        std::cout << std::endl;

        delete[] str_buffer;
        mpfr_clear(candidate);
        outfile.close();

        std::cout << "Prime candidates written to " << output_file << std::endl;
    }

    #ifdef USE_HIP
    // HIP kernel for generating random numbers
    __global__ void hip_generate_random(float* output, uint64_t count, uint64_t seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            // Simple LCG random number generator
            uint64_t lcg_seed = seed + idx;
            lcg_seed = (lcg_seed * 6364136223846793005ULL + 1442695040888963407ULL) % UINT64_MAX;
            output[idx] = static_cast<float>(lcg_seed) / UINT64_MAX;
        }
    }

    // Generate candidates using GPU acceleration
    void generate_candidates_hip() {
        std::ofstream outfile(output_file);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return;
        }

        std::cout << "Generating " << num_candidates << " prime candidates using HIP/ROCm..." << std::endl;

        // Allocate host and device memory
        const int BATCH_SIZE = 1024;
        float* h_random = new float[BATCH_SIZE];
        float* d_random;
        hipMalloc(&d_random, BATCH_SIZE * sizeof(float));

        mpfr_t candidate, range;
        mpfr_init2(candidate, MPFR_PRECISION);
        mpfr_init2(range, MPFR_PRECISION);
        char* str_buffer = new char[MPFR_PRECISION/2];

        // Calculate range
        mpfr_sub(range, x_upper, x_lower, MPFR_RNDN);

        // Process in batches
        for (uint64_t batch = 0; batch < (num_candidates + BATCH_SIZE - 1) / BATCH_SIZE; batch++) {
            uint64_t batch_start = batch * BATCH_SIZE;
            uint64_t batch_size = std::min(BATCH_SIZE, num_candidates - batch_start);

            // Launch kernel to generate random numbers
            dim3 blockSize(256);
            dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x);
            hip_generate_random<<<gridSize, blockSize>>>(d_random, batch_size, batch_start + std::time(nullptr));

            // Copy results back to host
            hipMemcpy(h_random, d_random, batch_size * sizeof(float), hipMemcpyDeviceToHost);

            // Process the random numbers
            for (uint64_t i = 0; i < batch_size; i++) {
                uint64_t global_idx = batch_start + i;

                // Convert random float to mpfr and scale to range
                mpfr_set_d(tmp1, h_random[i], MPFR_RNDN);
                mpfr_mul(tmp1, tmp1, range, MPFR_RNDN);
                mpfr_add(candidate, x_lower, tmp1, MPFR_RNDN);

                // Make it odd (potential prime)
                mpfr_floor(candidate);
                if (mpfr_get_ui(candidate, MPFR_RNDN) % 2 == 0) {
                    mpfr_add_ui(candidate, candidate, 1, MPFR_RNDN);
                }

                // Convert to string and write to file
                mpfr_get_str(str_buffer, nullptr, 10, 0, candidate, MPFR_RNDN);
                outfile << str_buffer << std::endl;
            }

            // Update progress
            std::cout << "\rGenerated " << std::min(batch_start + batch_size, num_candidates) << "/" << num_candidates
            << " candidates (" << std::fixed << std::setprecision(1)
            << (100.0 * std::min(batch_start + batch_size, num_candidates) / num_candidates) << "%)" << std::flush;
        }
        std::cout << std::endl;

        // Clean up
        delete[] h_random;
        delete[] str_buffer;
        hipFree(d_random);
        mpfr_clear(candidate);
        mpfr_clear(range);
        outfile.close();

        std::cout << "Prime candidates written to " << output_file << std::endl;
    }
    #endif

public:
    PrimeEstimator(uint64_t samples = 10000, double confidence = 0.95,
                   const std::string& outfile = "numbers.txt", uint64_t candidates = 100)
    : num_samples(samples), confidence_level(confidence),
    output_file(outfile), num_candidates(candidates),
    rng(std::random_device{}()),
    dist(0.0, 1.0) {

        // Initialize MPFR variables
        mpfr_init2(x_lower, MPFR_PRECISION);
        mpfr_init2(x_upper, MPFR_PRECISION);
        mpfr_init2(li_result, MPFR_PRECISION);
        mpfr_init2(log_x, MPFR_PRECISION);
        mpfr_init2(sqrt_x, MPFR_PRECISION);
        mpfr_init2(result, MPFR_PRECISION);
        mpfr_init2(tmp1, MPFR_PRECISION);
        mpfr_init2(tmp2, MPFR_PRECISION);
        mpfr_init2(tmp3, MPFR_PRECISION);
        mpfr_init2(correction, MPFR_PRECISION);
    }

    ~PrimeEstimator() {
        // Free MPFR variables
        mpfr_clear(x_lower);
        mpfr_clear(x_upper);
        mpfr_clear(li_result);
        mpfr_clear(log_x);
        mpfr_clear(sqrt_x);
        mpfr_clear(result);
        mpfr_clear(tmp1);
        mpfr_clear(tmp2);
        mpfr_clear(tmp3);
        mpfr_clear(correction);
    }

    // Estimate primes in exponential notation range (10^exp_lower, 10^exp_upper)
    std::pair<double, double> estimate_primes(double exp_lower, double exp_upper) {
        // Convert exponential notation to mpfr
        mpfr_set_d(tmp1, exp_lower, MPFR_RNDN);
        mpfr_set_d(tmp2, exp_upper, MPFR_RNDN);

        // Calculate 10^exp_lower and 10^exp_upper
        mpfr_set_ui(x_lower, 10, MPFR_RNDN);
        mpfr_pow(x_lower, x_lower, tmp1, MPFR_RNDN);

        mpfr_set_ui(x_upper, 10, MPFR_RNDN);
        mpfr_pow(x_upper, x_upper, tmp2, MPFR_RNDN);

        // Use Monte Carlo sampling with multithreading
        std::vector<std::pair<double, double>> all_results;
        std::vector<std::thread> threads;

        // Reset progress counter
        progress_counter.store(0);
        total_tasks.store(num_samples);

        // Calculate batch size per thread
        uint64_t batch_size = num_samples / NUM_THREADS;
        uint64_t remainder = num_samples % NUM_THREADS;

        // Start progress reporting thread
        std::atomic<bool> stop_progress{false};
        std::thread progress_thread([&]() {
            while (!stop_progress) {
                uint64_t current = progress_counter.load();
                uint64_t total = total_tasks.load();

                if (total > 0) {
                    double percent = (100.0 * current) / total;
                    std::cout << "\rProgress: " << current << "/" << total
                    << " (" << std::fixed << std::setprecision(2) << percent << "%)" << std::flush;
                }

                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

        // Launch worker threads
        for (int t = 0; t < NUM_THREADS; t++) {
            uint64_t start_idx = t * batch_size;
            uint64_t end_idx = (t + 1) * batch_size;

            // Add remainder to last thread
            if (t == NUM_THREADS - 1) {
                end_idx += remainder;
            }

            threads.emplace_back(&PrimeEstimator::process_range, this, start_idx, end_idx, std::ref(all_results));
        }

        // Wait for threads to finish
        for (auto& thread : threads) {
            thread.join();
        }

        // Stop progress reporting
        stop_progress = true;
        progress_thread.join();
        std::cout << std::endl;

        // Compute aggregate results
        double total_count = 0.0;
        double avg_density = 0.0;

        for (const auto& result : all_results) {
            total_count += result.first;
            avg_density += result.second;
        }

        avg_density /= all_results.size();

        // Calculate confidence interval
        double std_dev = 0.0;
        for (const auto& result : all_results) {
            std_dev += (result.second - avg_density) * (result.second - avg_density);
        }
        std_dev = std::sqrt(std_dev / all_results.size());

        // Z-score for the confidence level (e.g., 1.96 for 95% confidence)
        double z_score = 1.96;  // 95% confidence
        if (confidence_level > 0.99) z_score = 2.576;
        else if (confidence_level > 0.98) z_score = 2.33;
        else if (confidence_level > 0.95) z_score = 2.05;
        else if (confidence_level > 0.90) z_score = 1.645;

        double margin_of_error = z_score * std_dev / std::sqrt(all_results.size());

        std::cout << "Estimated prime count: " << std::scientific << total_count << std::endl;
        std::cout << "Prime density: " << avg_density << " ± " << margin_of_error << std::endl;

        // Generate prime candidates for primality testing
        #ifdef USE_HIP
        generate_candidates_hip();
        #else
        generate_candidates();
        #endif

        return {total_count, avg_density};
    }

    // In the estimate_ultra_large_primes function, more careful handling:
    std::pair<double, double> estimate_ultra_large_primes(double nested_exp) {
        // More gradual approach to calculating the range
        double log_order;

        if (nested_exp > 50) {
            // Use logarithm properties for very large exponents
            log_order = nested_exp * log(10);
        } else {
            log_order = std::pow(10, nested_exp);
        }

        // Use a safer range calculation
        double exp_lower = log_order * 0.9999;
        double exp_upper = log_order * 1.0001;

        // Add safety checks before division operations
        return estimate_primes(exp_lower, exp_upper);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <exponent> [num_samples] [num_candidates] [output_file]" << std::endl;
        std::cout << "Example for estimating primes around 10^10^9: " << argv[0] << " 9 100000 100 numbers.txt" << std::endl;
        return 1;
    }

    double nested_exp = std::stod(argv[1]);
    uint64_t num_samples = (argc > 2) ? std::stoull(argv[2]) : 10000;
    uint64_t num_candidates = (argc > 3) ? std::stoull(argv[3]) : 100;
    std::string output_file = (argc > 4) ? argv[4] : "numbers.txt";

    std::cout << "Estimating primes around 10^10^" << nested_exp << " using "
    << num_samples << " samples with " << NUM_THREADS << " threads" << std::endl;
    std::cout << "Will generate " << num_candidates << " prime candidates to " << output_file << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create estimator and run
    PrimeEstimator estimator(num_samples, 0.95, output_file, num_candidates);
    auto [prime_count, prime_density] = estimator.estimate_ultra_large_primes(nested_exp);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Estimation completed in " << duration.count() / 1000.0 << " seconds" << std::endl;

    return 0;
}
