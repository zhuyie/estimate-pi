#include <cstdio>
#include <cstdlib>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#endif

#include <vector>
#include <random>
#include <chrono>
using namespace std;
using namespace std::chrono;

static const char* kernelSource = R"(
#define MT19937_FLOAT_MULTI 2.3283064365386962890625e-10f

#define MT19937_N 624
#define MT19937_M 397
#define MT19937_MATRIX_A 0x9908b0df   /* constant vector a */
#define MT19937_UPPER_MASK 0x80000000 /* most significant w-r bits */
#define MT19937_LOWER_MASK 0x7fffffff /* least significant r bits */

/**
State of MT19937 RNG.
*/
typedef struct{
	uint mt[MT19937_N]; /* the array for the state vector  */
	int mti;
} mt19937_state;

/**
Generates a random 32-bit unsigned integer using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_uint(state) _mt19937_uint(&state)
uint _mt19937_uint(mt19937_state* state){
    uint y;
    uint mag01[2]={0x0, MT19937_MATRIX_A};
    /* mag01[x] = x * MT19937_MATRIX_A  for x=0,1 */
	
	if(state->mti<MT19937_N-MT19937_M){
		y = (state->mt[state->mti]&MT19937_UPPER_MASK)|(state->mt[state->mti+1]&MT19937_LOWER_MASK);
		state->mt[state->mti] = state->mt[state->mti+MT19937_M] ^ (y >> 1) ^ mag01[y & 0x1];
	}
	else if(state->mti<MT19937_N-1){
		y = (state->mt[state->mti]&MT19937_UPPER_MASK)|(state->mt[state->mti+1]&MT19937_LOWER_MASK);
		state->mt[state->mti] = state->mt[state->mti+(MT19937_M-MT19937_N)] ^ (y >> 1) ^ mag01[y & 0x1];
	}
	else{
        y = (state->mt[MT19937_N-1]&MT19937_UPPER_MASK)|(state->mt[0]&MT19937_LOWER_MASK);
        state->mt[MT19937_N-1] = state->mt[MT19937_M-1] ^ (y >> 1) ^ mag01[y & 0x1];
        state->mti = 0;
	}
    y = state->mt[state->mti++];
		
    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}

/**
Seeds MT19937 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mt19937_seed(mt19937_state* state, uint s){
    state->mt[0]= s;
	uint mti;
    for (mti=1; mti<MT19937_N; mti++) {
        state->mt[mti] = 1812433253 * (state->mt[mti-1] ^ (state->mt[mti-1] >> 30)) + mti;
		
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt19937[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
    }
	state->mti=mti;
}

/**
Generates a random float using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_float(state) (mt19937_uint(state)*MT19937_FLOAT_MULTI)


__kernel void pi(uint iters, __global uint* seed, __global uint* res)
{
    uint gid = get_global_id(0);
    mt19937_state state;
    mt19937_seed(&state, seed[gid]);
    uint cnt = 0;
    for (uint i = 0; i < iters; i++)
    {
        float a = mt19937_float(state);
        float b = mt19937_float(state);
        if (a * a + b * b <= 1.0f)
        {
            cnt++;
        }
    }
    res[gid] = cnt;
}
)";

#define CL_CHECK_RESULT(res, msg) \
do {                              \
    if (!(res))                   \
    {                             \
        fprintf(stderr, msg);     \
        return EXIT_FAILURE;      \
    }                             \
} while(0)

#define CL_CHECK_SUCCESS(res, msg) \
do {                               \
    if ((res) != CL_SUCCESS)       \
    {                              \
        fprintf(stderr, msg);      \
        return EXIT_FAILURE;       \
    }                              \
} while(0)

#define N_THREADS        1000*100
#define ITERS_PER_THREAD 10000

int main(int argc, char* argv[])
{
    int err;

    const int MAX_DEVICES = 4;
    cl_device_id deviceIDs[MAX_DEVICES] = { 0 };
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, MAX_DEVICES, deviceIDs, &numDevices);
    CL_CHECK_SUCCESS(err, "Error: Failed to get device ID!\n");
    if (numDevices == 0)
    {
        fprintf(stderr, "Error: no GPU found\n");
        return EXIT_FAILURE;
    }
    for (cl_uint i = 0; i < numDevices; i++)
    {
        char deviceName[101] = { 0 };
        clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 100, deviceName, NULL);
        fprintf(stdout, "Device_%d: %s\n", i+1, deviceName);

        char deviceVersion[101] = { 0 };
        clGetDeviceInfo(deviceIDs[i], CL_DEVICE_VERSION, 100, deviceVersion, NULL);
        fprintf(stdout, "    Hardware version: %s\n", deviceVersion);

        char driverVersion[101] = { 0 };
        clGetDeviceInfo(deviceIDs[i], CL_DRIVER_VERSION, 100, driverVersion, NULL);
        fprintf(stdout, "    Software version: %s\n", driverVersion);
    }
    fprintf(stdout, "\n");

    int device_index = 0;
    if (argc > 1)
    {
        device_index = atoi(argv[1]) - 1;
        if (device_index < 0 || device_index >= numDevices)
        {
            fprintf(stderr, "Invalid device_index!\n");
            return EXIT_FAILURE;
        }
    }
    fprintf(stdout, "Selected Device: Device_%d\n\n", device_index + 1);
    cl_device_id device = deviceIDs[device_index];

    // Create a compute context
    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    CL_CHECK_RESULT(context, "Error: Failed to create a compute context!\n");
    
    // Create a command queue
    cl_command_queue commands = clCreateCommandQueue(context, device, 0, &err);
    CL_CHECK_RESULT(commands, "Error: Failed to create a command queue!\n");
    
    // Create the compute program from the source buffer
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    CL_CHECK_RESULT(program, "Error: Failed to create compute program!\n");
    
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to build program!\n");
    
    // Create the compute kernel
    cl_kernel kernel = clCreateKernel(program, "pi", &err);
    CL_CHECK_RESULT(kernel, "Error: Failed to create compute kernel!\n");

    auto start = system_clock::now();

    // Prepare input && output buffer
    cl_mem seeds = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(cl_uint) * N_THREADS, NULL, NULL);
    cl_mem results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * N_THREADS, NULL, NULL);
    CL_CHECK_RESULT(seeds && results, "Error: Failed to allocate device memory!\n");
    vector<cl_uint> host_seeds(N_THREADS);
    mt19937 rnd;
    for (int i = 0; i < N_THREADS; i++)
    {
        host_seeds[i] = rnd();
    }
    err = clEnqueueWriteBuffer(commands, seeds, CL_TRUE, 0, sizeof(cl_uint) * N_THREADS, &host_seeds[0], 0, NULL, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to write to input buffer!\n");

    // Set the arguments to our compute kernel
    cl_uint iters = ITERS_PER_THREAD;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_uint), &iters);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &seeds);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &results);
    CL_CHECK_SUCCESS(err, "Error: Failed to set kernel arguments!\n");

    // Execute the kernel
    size_t global_workitems = N_THREADS;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_workitems, NULL, 0, NULL, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to execute kernel!\n");

    // Blocks until commands have completed
    clFinish(commands);

    // Read back the results from the device
    vector<cl_uint> host_results(N_THREADS);
    err = clEnqueueReadBuffer(commands, results, CL_TRUE, 0, sizeof(cl_uint) * N_THREADS, &host_results[0], 0, NULL, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to read output buffer!\n");
    
    // Calculate pi
    cl_ulong total = 0;
    for (cl_uint threadCount : host_results) 
    {
        total += threadCount;
    }
    double pi = static_cast<double>(total) / (ITERS_PER_THREAD * N_THREADS) * 4;
    double pi_true = acos(-1.0);  // true value of pi
    double error = abs(pi - pi_true) / pi_true * 100;
    
    auto duration = duration_cast<microseconds>(system_clock::now() - start);
    
    fprintf(stdout, "threads = %d\n", N_THREADS);
    fprintf(stdout, "iterates = %d\n", ITERS_PER_THREAD);
    fprintf(stdout, "samples = %lld\n", (long long)(ITERS_PER_THREAD * N_THREADS));
    fprintf(stdout, "duration = %.2fms\n", duration.count()/1000.0);
    fprintf(stdout, "pi = %f (%f%% error)\n", pi, error);

    // Shutdown and cleanup
    clReleaseMemObject(seeds);
    clReleaseMemObject(results);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
