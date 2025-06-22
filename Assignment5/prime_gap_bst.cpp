#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <limits>
#include <algorithm>
#include <chrono>

using namespace std;

// Binary tree node structure
struct TreeNode {
    long long int data;
    TreeNode* left;
    TreeNode* right;
    TreeNode(long long int val) : data(val), left(nullptr), right(nullptr) {}
};

// Function to check if a number is prime
bool isPrime(long long int num) {
    if (num <= 1)
        return false;
    if (num <= 3)
        return true;
    if (num % 2 == 0 || num % 3 == 0)
        return false;
    for (long long int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0)
            return false;
    }
    return true;
}

// Function to insert primes into BST using recursive midpoint search
void insertPrimes(TreeNode*& root, long long int n, long long int m) {
    if (n > m)
        return;

    long long int mid = (n + m) / 2;
    if ((mid - n) % 5000000 == 0) {
        #pragma omp critical
        cout << "Processing: " << mid << endl;
    }

    if (isPrime(mid)) {
        root = new TreeNode(mid);

        #pragma omp task shared(root)
        insertPrimes(root->left, n, mid - 1);

        #pragma omp task shared(root)
        insertPrimes(root->right, mid + 1, m);
    } else {
        long long int left = mid - 1, right = mid + 1;

        while (left >= n || right <= m) {
            if (left >= n && isPrime(left)) {
                root = new TreeNode(left);
                #pragma omp task shared(root)
                insertPrimes(root->left, n, left - 1);
                #pragma omp task shared(root)
                insertPrimes(root->right, left + 1, m);
                return;
            }
            if (right <= m && isPrime(right)) {
                root = new TreeNode(right);
                #pragma omp task shared(root)
                insertPrimes(root->left, n, right - 1);
                #pragma omp task shared(root)
                insertPrimes(root->right, right + 1, m);
                return;
            }
            left--;
            right++;
        }
    }
}

// In-order traversal to collect primes into a vector
void inorderCollect(TreeNode* root, vector<long long>& primes) {
    if (!root) return;
    inorderCollect(root->left, primes);
    primes.push_back(root->data);
    inorderCollect(root->right, primes);
}

int main() {
    long long int n, m;
    cout << "Enter the range [n, m]: ";
    cin >> n >> m;

    if (n >= m || n < 0 || m > numeric_limits<long long>::max()) {
        cerr << "Invalid range. Ensure n < m and values are within limits." << endl;
        return 1;
    }

    double start_time = omp_get_wtime();

    TreeNode* root = nullptr;

    #pragma omp parallel
    {
        #pragma omp single
        insertPrimes(root, n, m);
    }

    vector<long long> primes;
    inorderCollect(root, primes);

    if (primes.size() < 2) {
        cout << "Not enough prime numbers found in the given range." << endl;
        return 0;
    }

    // Compute max prime gap
    long long max_gap = 0;
    long long gap_start = 0, gap_end = 0;

    #pragma omp parallel for
    for (int i = 1; i < primes.size(); ++i) {
        long long gap = primes[i] - primes[i - 1];
        #pragma omp critical
        {
            if (gap > max_gap) {
                max_gap = gap;
                gap_start = primes[i - 1];
                gap_end = primes[i];
            }
        }
    }

    double end_time = omp_get_wtime();

    cout << "For n = " << n << ", m = " << m << ", the largest prime gap is between "
         << gap_start << " and " << gap_end << ", which is " << max_gap << "." << endl;
    cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;

    return 0;
}