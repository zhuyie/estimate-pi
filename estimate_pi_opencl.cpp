#include <cstdio>
#include <cstdlib>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <memory>
using namespace std;
using namespace std::chrono;

class CLSource
{
    std::string data_;
public:
    CLSource()
    {
    }
    const std::string& data() const
    {
        return data_;
    }
    bool load(const std::string& filename)
    {
        std::ifstream f(filename);
        if (!f.good())
            return false;
        f.seekg(0, std::ios::end);
        size_t size = f.tellg();
        data_.resize(size);
        f.seekg(0);
        f.read(&data_[0], size);
        if (data_.empty() || data_.back() != '\n')
            data_.append(1, '\n');
        return true;
    }
    int resolveInclude(const std::string& name, const CLSource& inc)
    {
        int n = 0;
        std::string incStr0 = "#include <" + name + ">";
        std::string incStr1 = "#include \"" + name + "\"";
        for (;;)
        {
            auto startPos = data_.find(incStr0);
            if (startPos != std::string::npos)
            {
                data_.replace(startPos, incStr0.length(), inc.data());
                n++;
                continue;
            }
            startPos = data_.find(incStr1);
            if (startPos != std::string::npos)
            {
                data_.replace(startPos, incStr1.length(), inc.data());
                n++;
                continue;
            }
            break;
        }
        return n;
    }
};

static bool loadSource(const std::string& dir, CLSource& src)
{
    CLSource mt19937;
    if (!mt19937.load(dir + "/3rdparty/RandomCL/generators/mt19937.cl"))
        return false;

    CLSource tinymt32, tinymt, tinymt32_jump_table, tinymt32def;
    if (!tinymt32.load(dir + "/3rdparty/RandomCL/generators/TinyMT/tinymt32_jump.clh"))
        return false;
    if (!tinymt.load(dir + "/3rdparty/RandomCL/generators/TinyMT/tinymt.clh"))
        return false;
    if (!tinymt32_jump_table.load(dir + "/3rdparty/RandomCL/generators/TinyMT/tinymt32_jump_table.clh"))
        return false;
    if (!tinymt32def.load(dir + "/3rdparty/RandomCL/generators/TinyMT/tinymt32def.h"))
        return false;
    tinymt32.resolveInclude("tinymt.clh", tinymt);
    tinymt32.resolveInclude("tinymt32_jump_table.clh", tinymt32_jump_table);
    tinymt32.resolveInclude("tinymt32def.h", tinymt32def);

    if (!src.load(dir + "/pi.cl"))
        return false;
    src.resolveInclude("mt19937.cl", mt19937);
    src.resolveInclude("tinymt32_jump.clh", tinymt32);

    return true;
}

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
    CLSource src;
    if (!loadSource("..", src))
    {
        fprintf(stderr, "Error: Can not load source\n");
        return EXIT_FAILURE;
    }

    int err;
    const int MAX_PLATFORMS = 2;
    const int MAX_DEVICES = 8;
    cl_platform_id platformIDs[MAX_PLATFORMS] = {0};
    cl_device_id deviceIDs[MAX_DEVICES] = {0};
    cl_uint numPlatforms = 0;
    cl_uint numDevices = 0;
    err = clGetPlatformIDs(MAX_PLATFORMS, platformIDs, &numPlatforms);
    CL_CHECK_SUCCESS(err, "Error: Failed to get platform ID!\n");
    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        cl_uint num = 0;
        err = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES-numDevices, deviceIDs+numDevices, &num);
        CL_CHECK_SUCCESS(err, "Error: Failed to get device ID!\n");
        numDevices += num;
    }
    if (numDevices == 0)
    {
        fprintf(stderr, "Error: no GPU found\n");
        return EXIT_FAILURE;
    }
    for (cl_uint i = 0; i < numDevices; i++)
    {
        char deviceName[101] = {0};
        clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 100, deviceName, NULL);
        fprintf(stdout, "Device_%d: %s\n", i + 1, deviceName);

        char deviceVersion[101] = {0};
        clGetDeviceInfo(deviceIDs[i], CL_DEVICE_VERSION, 100, deviceVersion, NULL);
        fprintf(stdout, "    Hardware version: %s\n", deviceVersion);

        char driverVersion[101] = {0};
        clGetDeviceInfo(deviceIDs[i], CL_DRIVER_VERSION, 100, driverVersion, NULL);
        fprintf(stdout, "    Software version: %s\n", driverVersion);
    }
    fprintf(stdout, "\n");

    int device_index = 0;
    if (argc > 1)
    {
        device_index = atoi(argv[1]) - 1;
        if (device_index < 0 || device_index >= (int)numDevices)
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
    const char* strings[1] = { src.data().c_str() };
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)strings, NULL, &err);
    CL_CHECK_RESULT(program, "Error: Failed to create compute program!\n");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "%s\n", buffer);
    }
    CL_CHECK_SUCCESS(err, "Error: Failed to build program!\n");

    // Create the compute kernel
    cl_kernel kernel = clCreateKernel(program, "pi_v2", &err);
    CL_CHECK_RESULT(kernel, "Error: Failed to create compute kernel!\n");

    auto start = system_clock::now();

    size_t max_workgroup_size = 0;
    cl_uint max_workitem_dims = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to query CL_DEVICE_MAX_WORK_GROUP_SIZE!\n");
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_workitem_dims, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to query CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS!\n");
    unique_ptr<size_t[]> max_workitem_sizes(new size_t[max_workitem_dims]);
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*max_workitem_dims, max_workitem_sizes.get(), NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to query CL_DEVICE_MAX_WORK_ITEM_SIZES!\n");
    // Calculate global_work_size/local_work_size
    size_t local_work_size = std::min(max_workgroup_size, max_workitem_sizes[0]);
    size_t global_work_size = ((N_THREADS - 1) / local_work_size + 1) * local_work_size;

    // Prepare input && output buffer
    cl_mem results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * global_work_size, NULL, NULL);
    CL_CHECK_RESULT(results, "Error: Failed to allocate device memory!\n");

    // Set the arguments to our compute kernel
    cl_uint iters = ITERS_PER_THREAD;
    cl_uint seed = 42;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_uint), &iters);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &seed);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &results);
    CL_CHECK_SUCCESS(err, "Error: Failed to set kernel arguments!\n");

    // Execute the kernel
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to execute kernel!\n");

    // Blocks until commands have completed
    clFinish(commands);

    // Read back the results from the device
    vector<cl_uint> host_results(global_work_size);
    err = clEnqueueReadBuffer(commands, results, CL_TRUE, 0, sizeof(cl_uint) * global_work_size, &host_results[0], 0, NULL, NULL);
    CL_CHECK_SUCCESS(err, "Error: Failed to read output buffer!\n");

    // Calculate pi
    cl_ulong total = 0;
    for (cl_uint threadCount : host_results) 
    {
        total += threadCount;
    }
    double pi = static_cast<double>(total) / (ITERS_PER_THREAD * global_work_size) * 4;
    double pi_true = acos(-1.0);  // true value of pi
    double error = abs(pi - pi_true) / pi_true * 100;

    auto duration = duration_cast<microseconds>(system_clock::now() - start);

    fprintf(stdout, "local_work_size = %d\n", (unsigned int)local_work_size);
    fprintf(stdout, "global_work_size = %d\n", (unsigned int)global_work_size);
    fprintf(stdout, "iterates = %d\n", ITERS_PER_THREAD);
    fprintf(stdout, "samples = %lld (%lld required)\n", 
        (long long)(ITERS_PER_THREAD * global_work_size), (long long)(ITERS_PER_THREAD * N_THREADS));
    fprintf(stdout, "duration = %.2fms\n", duration.count()/1000.0);
    fprintf(stdout, "pi = %f (%f%% error)\n", pi, error);
    fprintf(stdout, "\n");

    // Shutdown and cleanup
    clReleaseMemObject(results);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
