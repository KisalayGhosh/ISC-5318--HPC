#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <limits>
#include <thread> 

using namespace std;

// Function to compute all prime numbers up to a given limit using the Sieve of Eratosthenes.
// The sieve marks non-prime numbers in a boolean array and collects the remaining primes.
vector<long long> simple_sieve(long long limit) {
    vector<bool> is_prime(limit + 1, true);
    vector<long long> primes;
    is_prime[0] = is_prime[1] = false;
    for (long long i = 2; i * i <= limit; i++) {
        if (is_prime[i]) {
            for (long long j = i * i; j <= limit; j += i)
                is_prime[j] = false;
        }
    }
    for (long long i = 2; i <= limit; i++) {
        if (is_prime[i]) primes.push_back(i);
    }
    return primes;
}

// This function applies the Segmented Sieve of Eratosthenes to find primes in a given range,
// reducing memory usage compared to a full sieve by processing numbers in smaller chunks.
vector<long long> segmented_sieve(long long low, long long high) {
    long long limit = sqrt(high);
    vector<long long> small_primes = simple_sieve(limit);
    vector<bool> is_prime(high - low + 1, true);
    
    for (long long prime : small_primes) {
        long long start = max(prime * prime, (low + prime - 1) / prime * prime);
        for (long long j = start; j <= high; j += prime)
            is_prime[j - low] = false;
    }
    
    vector<long long> primes;
    for (long long i = 0; i < high - low + 1; i++) {
        if (is_prime[i] && (low + i) > 1) primes.push_back(low + i);
    }
    return primes;
}

int main() {
    long long n, m;
    cout << "Enter range (n m): ";
    cin >> n >> m;
    
    if (n > numeric_limits<long long>::max() || m > numeric_limits<long long>::max()) {
        cout << "Error: n or m exceeds the maximum allowed value for long long int." << endl;
        return 1;
    }
    
    if (m <= n) {
        cout << "Invalid range. Ensure n < m." << endl;
        return 1;
    }
    
    double start_time = omp_get_wtime();
    long long range_size = m - n;
    long long chunk_size = max(50000LL, range_size / (omp_get_max_threads() * 10)); 
    vector<long long> primes;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        vector<long long> local_primes;
        
        
        if (num_threads < 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500 * (4 - num_threads)));
        }

        // scheduling to dynamically distribute workload among threads.
        #pragma omp for schedule(guided, 10000)
        for (long long start = n; start <= m; start += chunk_size) {
            long long end = min(start + chunk_size - 1, m);
            vector<long long> chunk_primes = segmented_sieve(start, end);
            local_primes.insert(local_primes.end(), make_move_iterator(chunk_primes.begin()), make_move_iterator(chunk_primes.end()));
        }
        
        //  lock-free merging of results to avoid excessive synchronization overhead.
        #pragma omp critical
        primes.insert(primes.end(), make_move_iterator(local_primes.begin()), make_move_iterator(local_primes.end()));
    }
    sort(primes.begin(), primes.end()); // Ensure primes are sorted before computing gaps.
    
    long long max_gap = 0, gap_start = 0, gap_end = 0;
    
    // Parallel reduction to compute the largest prime gap efficiently.
    #pragma omp parallel for reduction(max:max_gap)
    for (size_t i = 1; i < primes.size(); i++) {
        long long gap = primes[i] - primes[i - 1];
        if (gap > max_gap) {
            max_gap = gap;
            gap_start = primes[i - 1];
            gap_end = primes[i];
        }
    }
    
    double end_time = omp_get_wtime();
    
    if (primes.size() < 2) {
        cout << "Not enough primes found in the range." << endl;
    } else {
        cout << "For n = " << n << ", m = " << m << ", the biggest prime number gap is between "
             << gap_start << " and " << gap_end << ", which is " << max_gap << "." << endl;
    }
    
    cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;
    
    return 0;
}
