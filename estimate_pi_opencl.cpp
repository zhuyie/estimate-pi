#include <cstdio>
#include <cstdlib>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include "GPAInterfaceLoader.h"
#define GPA_ENABLED 1
#endif

#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <memory>
using namespace std;
using namespace std::chrono;

//------------------------------------------------------------------------------

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
        // some file may contains trailing '\0'
        size_t sz = strlen(&data_[0]);
        if (sz < size)
            data_.resize(sz);
        // make sure we end with a '\n'
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

//------------------------------------------------------------------------------

#if GPA_ENABLED
GPAApiManager* GPAApiManager::m_pGpaApiManager = nullptr;
GPAFuncTableInfo* g_pFuncTableInfo = nullptr;
GPAFunctionTable* pGpaFunctionTable = nullptr;
GPA_ContextId gpaContextId = nullptr;
GPA_SessionId gpaSessionId = nullptr;
GPA_CommandListId gpaCommandListId = nullptr;
bool gpaInitOK = false;
#endif

static bool GPA_Init(cl_command_queue context, unsigned int &passRequired)
{
#if GPA_ENABLED
    GPA_Status status;
    status = GPAApiManager::Instance()->LoadApi(GPA_API_OPENCL);
    if (status != GPA_STATUS_OK)
        return false;
    pGpaFunctionTable = GPAApiManager::Instance()->GetFunctionTable(GPA_API_OPENCL);
    if (nullptr == pGpaFunctionTable)
        return false;
    status = pGpaFunctionTable->GPA_Initialize(GPA_INITIALIZE_DEFAULT_BIT);
    if (status != GPA_STATUS_OK)
        return false;
    status = pGpaFunctionTable->GPA_OpenContext(context, GPA_OPENCONTEXT_DEFAULT_BIT, &gpaContextId);
    if (status != GPA_STATUS_OK)
        return false;

    status = pGpaFunctionTable->GPA_CreateSession(gpaContextId, GPA_SESSION_SAMPLE_TYPE_DISCRETE_COUNTER, &gpaSessionId);
    if (status != GPA_STATUS_OK)
        return false;
    status = pGpaFunctionTable->GPA_EnableAllCounters(gpaSessionId);
    if (status != GPA_STATUS_OK)
        return false;
    status = pGpaFunctionTable->GPA_GetPassCount(gpaSessionId, &passRequired);
    if (status != GPA_STATUS_OK)
        return false;
    status = pGpaFunctionTable->GPA_BeginSession(gpaSessionId);
    if (status != GPA_STATUS_OK)
        return false;

    gpaInitOK = true;
    return true;
#else
    return false;
#endif
}

static bool GPA_BeginPass(unsigned int pass)
{
#if GPA_ENABLED
    if (!gpaInitOK)
        return true;

    GPA_Status status = pGpaFunctionTable->GPA_BeginCommandList(
        gpaSessionId, pass, GPA_NULL_COMMAND_LIST, GPA_COMMAND_LIST_NONE, &gpaCommandListId);
    if (status != GPA_STATUS_OK)
        return false;

    status = pGpaFunctionTable->GPA_BeginSample(0, gpaCommandListId);
    if (status != GPA_STATUS_OK)
        return false;

    return true;
#else
    return true;
#endif
}

static bool GPA_EndPass(unsigned int pass)
{
#if GPA_ENABLED
    if (!gpaInitOK)
        return true;

    if (gpaCommandListId)
    {
        GPA_Status status;
        status = pGpaFunctionTable->GPA_EndSample(gpaCommandListId);
        if (status != GPA_STATUS_OK)
            return false;

        status = pGpaFunctionTable->GPA_EndCommandList(gpaCommandListId);
        if (status != GPA_STATUS_OK)
            return false;

        for (;;)
        {
            status = pGpaFunctionTable->GPA_IsPassComplete(gpaSessionId, pass);
            if (status == GPA_STATUS_OK)
                break;
        }

        gpaCommandListId = nullptr;
    }

    return true;
#else
    return true;
#endif
}

static void GPA_Uninit()
{
#if GPA_ENABLED
    GPA_Status status;
    if (pGpaFunctionTable)
    {
        if (gpaSessionId)
        {
            status = pGpaFunctionTable->GPA_EndSession(gpaSessionId);
            if (status != GPA_STATUS_OK)
                fprintf(stderr, "GPA_EndSession failed, status=%d\n", status);

            size_t resultSize = 0;
            status = pGpaFunctionTable->GPA_GetSampleResultSize(gpaSessionId, 0, &resultSize);
            if (status != GPA_STATUS_OK || resultSize == 0)
                fprintf(stderr, "GPA_GetSampleResultSize failed, status=%d, resultSize=%u\n", 
                    status, (unsigned int)resultSize);

            vector<unsigned char> resultData(resultSize);
            status = pGpaFunctionTable->GPA_GetSampleResult(gpaSessionId, 0, resultSize, &resultData[0]);
            if (status != GPA_STATUS_OK)
                fprintf(stderr, "GPA_GetSampleResult failed, status=%d\n", status);

            fprintf(stdout, "-------- GPA RESULT --------\n");
            gpa_uint32 numCounters = 0;
            pGpaFunctionTable->GPA_GetNumEnabledCounters(gpaSessionId, &numCounters);
            for (gpa_uint32 i = 0; i < numCounters; i++)
            {
                gpa_uint32 enabledIndex = 0;
                pGpaFunctionTable->GPA_GetEnabledIndex(gpaSessionId, i, &enabledIndex);
                const char* name = nullptr;
                pGpaFunctionTable->GPA_GetCounterName(gpaContextId, enabledIndex, &name);
                GPA_Data_Type dt = GPA_DATA_TYPE_FLOAT64;
                pGpaFunctionTable->GPA_GetCounterDataType(gpaContextId, enabledIndex, &dt);
                if (dt == GPA_DATA_TYPE_FLOAT64)
                {
                    gpa_float64 val = *(reinterpret_cast<gpa_float64*>(&resultData[0]) + i);
                    fprintf(stdout, "%02u_%s: %.4f\n", i + 1, name, val);
                }
                else if (dt == GPA_DATA_TYPE_UINT64)
                {
                    gpa_uint64 val = *(reinterpret_cast<gpa_uint64*>(&resultData[0]) + i);
                    fprintf(stdout, "%02u_%s: %llu\n", i + 1, name, (unsigned long long)val);
                }
            }
            fprintf(stdout, "\n");
        }
        if (gpaContextId)
        {
            pGpaFunctionTable->GPA_CloseContext(gpaContextId);
            gpaContextId = nullptr;
        }
        pGpaFunctionTable->GPA_Destroy();
        pGpaFunctionTable = nullptr;
    }
    GPAApiManager::Instance()->UnloadApi(GPA_API_OPENCL);
#else
    // do nothing
#endif
}

//------------------------------------------------------------------------------

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
    int deviceIndex = 1;
    const char* kernelName = "pi_v2";
    bool profiling = true;
    if (argc >= 2)
        deviceIndex = atoi(argv[1]);
    if (argc >= 3)
        kernelName = argv[2];
    if (argc >= 4)
        profiling = (strcmp(argv[3], "1") == 0);
    fprintf(stdout, "device_index: %d\n", deviceIndex);
    fprintf(stdout, "kernel: %s\n", kernelName);
    fprintf(stdout, "profiling: %d\n", profiling ? 1 : 0);
    fprintf(stdout, "\n");

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

    if (deviceIndex <= 0 || deviceIndex > (int)numDevices)
    {
         fprintf(stderr, "Invalid device_index!\n");
         return EXIT_FAILURE;
    }
    fprintf(stdout, "Selected Device: Device_%d\n\n", deviceIndex);
    cl_device_id device = deviceIDs[deviceIndex - 1];

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
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    CL_CHECK_RESULT(kernel, "Error: Failed to create compute kernel!\n");

    unsigned int numPasses = 1;
    if (profiling)
    {
        if (GPA_Init(commands, numPasses))
        {
            fprintf(stdout, "GPA init OK, numPasses=%u\n\n", numPasses);
        }
        else
        {
            fprintf(stderr, "GPA init failed\n\n");
            numPasses = 1;
        }
    }

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

    for (unsigned int pass = 0; pass < numPasses; pass++)
    {
        if (!GPA_BeginPass(pass))
            fprintf(stderr, "GPA_BeginPass failed, pass=%u\n", pass);

        // Execute the kernel
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        CL_CHECK_SUCCESS(err, "Error: Failed to execute kernel!\n");

        if (!GPA_EndPass(pass))
            fprintf(stderr, "GPA_EndPass failed, pass=%u\n", pass);
    }

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
    fprintf(stdout, "duration = %.2fms\n", duration.count()/(1000.0*numPasses));
    fprintf(stdout, "pi = %f (%f%% error)\n", pi, error);
    fprintf(stdout, "\n");

    GPA_Uninit();

    // Shutdown and cleanup
    clReleaseMemObject(results);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
