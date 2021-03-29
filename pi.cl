#include <mt19937.cl>
#define KERNEL_PROGRAM
#include <tinymt32_jump.clh>

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__kernel
void pi_v1(uint iters, 
           uint seed, 
           __global uint* global_sum)
{
    const uint global_id = get_global_id(0);
    mt19937_state state;
    mt19937_seed(&state, wang_hash(global_id));
    uint sum = 0;
    for (uint i = 0; i < iters; i++)
    {
        float a = mt19937_float(state);
        float b = mt19937_float(state);
        if (a * a + b * b <= 1.0f)
        {
            sum++;
        }
    }
    global_sum[global_id] = sum;
}

__kernel
void pi_v2(uint iters,
           uint seed,
           __global uint* global_sum)
{
    tinymt32j_t tiny;
    tinymt32j_init_jump(&tiny, seed);
    uint sum = 0;
    for (uint i = 0; i < iters; i++)
    {
        float x = tinymt32j_single01(&tiny);
        float y = tinymt32j_single01(&tiny);
        if (x * x + y * y <= 1.0f) 
        {
            sum++;
        }
    }
    const size_t global_id = get_global_id(0);
    global_sum[global_id] = sum;
}
