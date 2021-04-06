/* Port from OpenCL version of TinyMT. */

/**
 * Copyright (C) 2013 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */

#ifndef __TINYMT32J_H__
#define __TINYMT32J_H__

typedef unsigned int uint;

/**
 * TinyMT32 structure for jump without parameters
 */
typedef struct TINYMT32J_T {
    uint s0;
    uint s1;
    uint s2;
    uint s3;
} tinymt32j_t;

#define TINYMT32J_MAT1 0x8f7011eeU
#define TINYMT32J_MAT2 0xfc78ff1fU
#define TINYMT32J_TMAT 0x3793fdffU

#define TINYMT32_JUMP_TABLE_SIZE 32

/**
 * Jump table contains pre-calculated jump polynomials.
 * This table supports sequential ids from 0 to 2<sup>32</sup>-1.
 */
const uint
tinymt32_jump_table[TINYMT32_JUMP_TABLE_SIZE][4] = {
    {0x4ef38e31,0xca64cb3e,0x58005925,0x50029072},
    {0x0db944dd,0xb5baf1a6,0x11190368,0x002e97f0},
    {0xac90fcc3,0x459849d2,0xa66f183f,0x51fa5216},
    {0x69e67e51,0xf16d8493,0x82a09356,0x6854e55d},
    {0xf647e755,0xcdface1f,0xda3b0222,0x590aebea},
    {0xd8c552ac,0xd4c99de1,0x5534a0d9,0x41e11626},
    {0x86d67c2c,0x7e4a30e9,0xe1cc666c,0x52c8d38d},
    {0x61c6371f,0xf7538850,0x00728662,0x0259f94f},
    {0x23f0f107,0x0d890068,0x1ba19dc9,0x654f5b29},
    {0x7aa1c52d,0x7bdabfce,0x68d36eb2,0x5c2fb73c},
    {0x223a1358,0x444e55c9,0xb8f9fceb,0x593a625c},
    {0x249949e2,0xf28a13f6,0x5770bc89,0x0d45e674},
    {0xa7347dc3,0x559bd9e0,0x9a21b9b1,0x574421e1},
    {0xc4565686,0xd9824575,0x278853b4,0x72ed8cde},
    {0x3553d677,0xb9721d05,0x79a24b89,0x56df35e9},
    {0x87d3a7e1,0xddcbc4ff,0xc413f22d,0x1a2f250d},
    {0xb7d5f3f0,0xfbf8abcb,0xb5cf72a0,0x22015702},
    {0xf8077f95,0x7f16d6eb,0x23574b64,0x5c06e4be},
    {0xbad2b02e,0xfccb2a6a,0xef9ab3c5,0x3441b10a},
    {0x9bb621ce,0xd59cb9c8,0x58870bb0,0x6273cdc4},
    {0x423514ef,0x88755f9e,0xa932d1f3,0x094ff35e},
    {0x77fcffdf,0xc6a1a5a3,0x4a3f4192,0x6f94cc6f},
    {0x175f1173,0x3f2e3836,0x606d1d5f,0x2c359461},
    {0xd2301c4a,0x87ae0696,0xcc5ff77f,0x21979d7a},
    {0xa3b01770,0x4c42e38a,0xefa5843c,0x509e623c},
    {0x8edd0470,0x5c110e0a,0x2dea653a,0x46cbfaea},
    {0x56c8b55b,0x21c4ae8d,0x3b70a522,0x2ddb453e},
    {0x2b3a0aba,0xd305e144,0x50907d57,0x0286e815},
    {0x88122803,0xec45f4cc,0x6201768d,0x592031bb},
    {0xbb30e1f1,0x34e35800,0xbed567bb,0x6872608f},
    {0x354be8b4,0x9475189a,0x1d657fd4,0x5761dc09},
    {0x82fc046e,0x97fd9c61,0x37cae8de,0x1c377678}
};

#define TINYMT32J_MIN_LOOP 8
#define TINYMT32J_PRE_LOOP 8

const int tinymt32j_sh0 = 1;
const int tinymt32j_sh1 = 10;
const int tinymt32j_sh8 = 8;
const uint tinymt32j_mask = 0x7fffffffU;
const uint tinymt32j_mat1 = TINYMT32J_MAT1;
const uint tinymt32j_mat2 = TINYMT32J_MAT2;
const uint tinymt32j_tmat = TINYMT32J_TMAT;
const uint tinymt32j_tmat_float = (TINYMT32J_TMAT >> 9) | 0x3f800000U;

/**
 * Addition of internal state
 * @param dest destination (changed)
 * @param src source (not changed)
 */
inline static void
tinymt32j_add(tinymt32j_t * dest, tinymt32j_t * src)
{
    dest->s0 ^= src->s0;
    dest->s1 ^= src->s1;
    dest->s2 ^= src->s2;
    dest->s3 ^= src->s3;
}

/**
 * State transition function
 * @param tiny internal state
 */
inline static void
tinymt32j_next_state(tinymt32j_t *tiny)
{
    uint x;
    uint y;

    y = tiny->s3;
    x = (tiny->s0 & tinymt32j_mask)
        ^ tiny->s1
        ^ tiny->s2;
    x ^= (x << tinymt32j_sh0);
    y ^= (y >> tinymt32j_sh0) ^ x;
    tiny->s0 = tiny->s1;
    tiny->s1 = tiny->s2;
    tiny->s2 = x ^ (y << tinymt32j_sh1);
    tiny->s3 = y;
    if (y & 1) {
	tiny->s1 ^= tinymt32j_mat1;
	tiny->s2 ^= tinymt32j_mat2;
    }
}

/**
 * Tempering function
 * @param tiny internal state
 * @return generated number
 */
inline static uint
tinymt32j_temper(tinymt32j_t *tiny)
{
    uint t0;
    uint t1;
    t0 = tiny->s3;
    t1 = tiny->s0
        + (tiny->s2 >> tinymt32j_sh8);
    t0 ^= t1;
    if (t1 & 1) {
	t0 ^= tinymt32j_tmat;
    }
    return t0;
}

/**
 * Tempering function
 * @param tiny internal state
 * @return generated number
 */
inline static uint
tinymt32j_uint32(tinymt32j_t *tiny)
{
    tinymt32j_next_state(tiny);
    return tinymt32j_temper(tiny);
}

/**
 * Tempering function
 * @param tiny internal state
 * @return generated number
 */
inline static float
tinymt32j_temper_float12(tinymt32j_t *tiny)
{
    uint t0;
    uint t1;
    t0 = tiny->s3;
    t1 = tiny->s0
        + (tiny->s2 >> tinymt32j_sh8);
    t0 ^= t1;
    if (t1 & 1) {
	t0 = (t0 >> 9) ^ tinymt32j_tmat_float;
    } else {
	t0 = (t0 >> 9) ^ 0x3f800000U;
    }
    return *reinterpret_cast<float*>(&t0);
}

/**
 * Tempering function
 * @param tiny internal state
 * @return generated number
 */
inline static float
tinymt32j_single12(tinymt32j_t *tiny)
{
    tinymt32j_next_state(tiny);
    return tinymt32j_temper_float12(tiny);
}

/**
 * Tempering function
 * @param tiny internal state
 * @return generated number
 */
inline static float
tinymt32j_single01(tinymt32j_t *tiny)
{
    return tinymt32j_single12(tiny) - 1.0f;
}

/**
 * Internal function.
 * This function certificate the period of 2^127-1.
 * @param tiny tinymt state vector.
 */
inline static void
tinymt32j_period_certification(tinymt32j_t * tiny)
{
    if ((tiny->s0 & tinymt32j_mask) == 0 &&
        tiny->s1 == 0 &&
        tiny->s2 == 0 &&
        tiny->s3 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'I';
        tiny->s2 = 'N';
        tiny->s3 = 'Y';
    }
}

/**
 * This function initializes the internal state array with a 32-bit
 * unsigned integer seed.
 * @param tiny tinymt state vector.
 * @param seed a 32-bit unsigned integer used as a seed.
 */
inline static void
tinymt32j_init_seed(tinymt32j_t *tiny, uint seed)
{
    uint status[4];
    status[0] = seed;
    status[1] = tinymt32j_mat1;
    status[2] = tinymt32j_mat2;
    status[3] = tinymt32j_tmat;
    for (int i = 1; i < TINYMT32J_MIN_LOOP; i++) {
        status[i & 3] ^= i + 1812433253U
            * (status[(i - 1) & 3]
               ^ (status[(i - 1) & 3] >> 30));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tiny->s2 = status[2];
    tiny->s3 = status[3];
    tinymt32j_period_certification(tiny);
    for (int i = 0; i < TINYMT32J_PRE_LOOP; i++) {
        tinymt32j_next_state(tiny);
    }
}

/**
 * jump using the jump polynomial.
 * This function is not as time consuming as calculating jump polynomial.
 * This function can use multiple time for the tinymt32j structure.
 * @param tiny tinymt32j structure, overwritten by new state after calling
 * this function.
 * @param jump_array the jump polynomial calculated by
 * tinymt32j_calculate_jump_polynomial.
 */
inline static void
tinymt32j_jump_by_array(tinymt32j_t *tiny,
			            const uint * jump_array)
{
    tinymt32j_t work_z;
    tinymt32j_t *work = &work_z;
    work->s0 = 0;
    work->s1 = 0;
    work->s2 = 0;
    work->s3 = 0;

    for (int i = 0; i < 4; i++) {
	uint x = jump_array[i];
	for (int i = 0; i < 32; i++) {
	    if ((x & 1) != 0) {
		tinymt32j_add(work, tiny);
	    }
	    tinymt32j_next_state(tiny);
	    x = x >> 1;
	}
    }
    *tiny = *work;
}

inline static void
tinymt32j_init_jump(tinymt32j_t *tiny, uint seed, uint gid)
{
    tinymt32j_init_seed(tiny, seed);
    //size_t gid = tinymt_get_sequential_id();
    for (int i = 0; (gid != 0) && (i < TINYMT32_JUMP_TABLE_SIZE); i++) {
	if ((gid & 1) != 0) {
	    tinymt32j_jump_by_array(tiny, tinymt32_jump_table[i]);
	}
	gid = gid >> 1;
    }
}

#endif /* EOF */
