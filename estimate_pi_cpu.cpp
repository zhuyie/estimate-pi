#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>
#include <vector>
using namespace std;
using namespace chrono;

static void usage()
{
    fprintf(stdout, "usage: estimate_pi_cpu num_threads num_samples\n");
    exit(1);
}

class RandomNumber
{
    mt19937 gen_;  // random number generator
    const float MT19937_FLOAT_MULTI = 2.3283064365386962890625e-10f; // (2^32-1)^-1
public:
    RandomNumber() : gen_()
    {
    }
    float operator() ()
    {
        return gen_() * MT19937_FLOAT_MULTI;
    }
    void seed(unsigned int z)
    {
        gen_.seed(z);
    }
};

union Counter
{
    int64_t value;
    char __padding[64];  // padding to cache line size
    Counter() : value(0) { }
};

void worker(int worker_id, int64_t samples, RandomNumber *rnd, Counter *in)
{
    rnd->seed(worker_id);  // seed the prng 
    for (int64_t i = 0; i < samples; i++)
    {
        float x = (*rnd)();
        float y = (*rnd)();
        if (x*x + y*y <= 1)
        {
            (in->value)++;
        }
    }
}

int main(int argc, char* argv[])
{
    int num_threads = 1;
    int64_t num_samples = 1000000000;
    if (argc >= 2)
    {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0 || num_threads > 32)
            usage();
    }
    if (argc >= 3)
    {
        num_samples = atoll(argv[2]);
        if (num_samples <= 0)
            usage();
    }

    auto start = system_clock::now();

    RandomNumber* rnd = new RandomNumber[num_threads];
    Counter* in_points = new Counter[num_threads];

    int64_t total_points = num_samples;
    int64_t circle_points = 0;
    if (num_threads > 1)
    {
        vector<thread> threads;
        for (int i = 0; i < num_threads; i++)
        {
            int64_t samples = num_samples / num_threads;
            if (i == num_threads - 1)
            {
                samples += num_samples - samples * num_threads;
            }
            threads.emplace_back(worker, i, samples, &(rnd[i]), &(in_points[i]));
        }
        for (int i = 0; i < num_threads; i++)
        {
            threads[i].join();
            circle_points += in_points[i].value;
        }
    }
    else
    {
        worker(0, total_points, &(rnd[0]), &(in_points[0]));
        circle_points = in_points[0].value;
    }

    double pi = 4 * circle_points / (double)total_points;
    double pi_true = acos(-1.0);  // true value of pi
    double error = abs(pi - pi_true) / pi_true * 100;
    
    auto duration = duration_cast<microseconds>(system_clock::now() - start);

    fprintf(stdout, "threads = %d\n", num_threads);
    fprintf(stdout, "samples = %lld\n", (long long)num_samples);
    fprintf(stdout, "duration = %.2fms\n", duration.count()/1000.0);
    fprintf(stdout, "pi = %f (%f%% error)\n", pi, error);
    fprintf(stdout, "\n");

    delete []rnd;
    delete []in_points;

    return 0;
}
