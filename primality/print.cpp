#include <iostream>
#include <gmpxx.h>
#include <thread>
#include <vector>
#include <chrono>
#include <future>
#include <numeric>

struct CalculationResult {
    mpz_class result;
    double duration;
    size_t digit_count;
    std::string first_20;
    std::string last_20;
};

CalculationResult calculate_power_single_thread(const mpz_class& base, const mpz_class& exponent) {
    auto start = std::chrono::high_resolution_clock::now();
    mpz_class result;
    mpz_pow_ui(result.get_mpz_t(), base.get_mpz_t(), exponent.get_ui());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    CalculationResult calc_result;
    calc_result.result = result;
    calc_result.duration = duration.count();
    calc_result.digit_count = mpz_sizeinbase(result.get_mpz_t(), 10) + 1;

    std::string result_str = result.get_str();
    if (result_str.length() > 20) {
        calc_result.first_20 = result_str.substr(0, 20);
        calc_result.last_20 = result_str.substr(result_str.length() - 20);
    } else {
        calc_result.first_20 = result_str;
        calc_result.last_20 = result_str;
    }
    return calc_result;
}

CalculationResult calculate_power_multi_thread_optimized(const mpz_class& base, const mpz_class& exponent) {
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = std::max(1u, num_threads); // Ensure at least one thread

    if (num_threads == 1) { // If only 1 thread, do sequential.
        return calculate_power_single_thread(base, exponent);
    }

    std::vector<std::future<mpz_class>> futures;
    std::vector<mpz_class> partial_exponents(num_threads);

    mpz_class partial_exponent = exponent / num_threads;
    mpz_class remainder = exponent % num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        partial_exponents[i] = partial_exponent;
        if (i == num_threads - 1) {
            partial_exponents[i] += remainder;
        }

        futures.push_back(std::async(std::launch::async, [base, exp = partial_exponents[i]]() {
            mpz_class result;
            mpz_pow_ui(result.get_mpz_t(), base.get_mpz_t(), exp.get_ui());
            return result;
        }));
    }

    std::vector<mpz_class> partial_results(num_threads);
    for (unsigned int i = 0; i < num_threads; ++i) {
        partial_results[i] = futures[i].get();
    }

    mpz_class result = std::accumulate(partial_results.begin(), partial_results.end(), mpz_class(1), std::multiplies<mpz_class>());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    CalculationResult calc_result;
    calc_result.result = result;
    calc_result.duration = duration.count();
    calc_result.digit_count = mpz_sizeinbase(result.get_mpz_t(), 10) + 1;

    std::string result_str = result.get_str();
    if (result_str.length() > 20) {
        calc_result.first_20 = result_str.substr(0, 20);
        calc_result.last_20 = result_str.substr(result_str.length() - 20);
    } else {
        calc_result.first_20 = result_str;
        calc_result.last_20 = result_str;
    }
    return calc_result;
}

int main() {
    mpz_class base = 2;
    mpz_class exponent = 332192810;

    // Single-threaded calculation (including I/O)
    auto start_single = std::chrono::high_resolution_clock::now(); // Start timing here
    CalculationResult single_result = calculate_power_single_thread(base, exponent);

    std::cout << "Single-threaded Calculation:" << std::endl;
    std::cout << "  Digit Count: " << single_result.digit_count << std::endl;
    std::cout << "  First 20 Digits: " << single_result.first_20 << std::endl;
    std::cout << "  Last 20 Digits: " << single_result.last_20 << std::endl;

    auto end_single = std::chrono::high_resolution_clock::now(); // End timing here
    std::chrono::duration<double> duration_single = end_single - start_single;

    std::cout << "  Total Duration: " << duration_single.count() << " seconds" << std::endl; // changed to total duration.

    // Multithreaded calculation (including I/O)
    auto start_multi = std::chrono::high_resolution_clock::now(); // Start timing here
    CalculationResult multi_result = calculate_power_multi_thread_optimized(base, exponent);

    std::cout << "\nMultithreaded Calculation (Optimized):" << std::endl;
    std::cout << "  Digit Count: " << multi_result.digit_count << std::endl;
    std::cout << "  First 20 Digits: " << multi_result.first_20 << std::endl;
    std::cout << "  Last 20 Digits: " << multi_result.last_20 << std::endl;

    auto end_multi = std::chrono::high_resolution_clock::now(); // End timing here
    std::chrono::duration<double> duration_multi = end_multi - start_multi;

    std::cout << "  Total Duration: " << duration_multi.count() << " seconds" << std::endl; // changed to total duration.

    //Verify Results
    if (single_result.result == multi_result.result) {
        std::cout << "\nResults Match." << std::endl;
    } else {
        std::cout << "\nResults do not match." << std::endl;
    }

    return 0;
}

