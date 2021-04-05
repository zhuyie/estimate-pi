#include <metal_stdlib>
using namespace metal;

#define FLOAT_MULTI 2.3283064365386962890625e-10f

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

kernel void pi(constant uint& iters [[buffer(0)]],
              constant uint& seed [[buffer(1)]],
              device uint* result [[buffer(2)]],
              uint index [[thread_position_in_grid]])
{
    uint r = wang_hash(seed + index);
    uint sum = 0;
    for (uint i = 0; i < iters; i++)
    {
        r = wang_hash(r);
        float x = r * FLOAT_MULTI;
        r = wang_hash(r);
        float y = r * FLOAT_MULTI;
        if (x * x + y * y <= 1.0f)
        {
            sum++;
        }
    }
    result[index] = sum;
}
