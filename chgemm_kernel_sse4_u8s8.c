#include "chgemm_kernel.h"

#include <smmintrin.h>

void chgemm_build_loc_sse4_u8s8(void *src, void *dst, int lds, int block_row, int block_col)
{
    int i, j;
    __m128i xmm1;

    for (i = 0; i < block_row; i++)
    {
        for (j = 0; j < block_col; j += 16)
        {
            xmm1 = _mm_load_si128((const __m128i*)(src + j));
            _mm_store_si128((__m128i*)(dst + j), xmm1);
        }
        src += lds;
        dst += 64;
    }
}

void chgemm_c_loc_clear_sse4_u8s8(void *c, size_t size)
{
    int i;
    __m128i xmm0 = _mm_setzero_si128();
    for (i = 0; i < size; i += 16)
    {
        _mm_store_si128((__m128i*)(c + i), xmm0);
    }
}

void chgemm_c_back_sse4_u8s8(int *cm, int *c, int ldc, int blk_m, int blk_n)
{
    int i, j;
    for (i = 0; i <= blk_n - 2; i += 2)
    {
        for (j = 0; j < blk_m; j++)
        {
            __m128i xmm0, xmm1;
            xmm0 = _mm_load_si128((__m128i*)cm);
            xmm1 = _mm_load_si128((__m128i*)(cm + 4));
            xmm0 = _mm_hadd_epi32(xmm0, xmm1);
            xmm0 = _mm_hadd_epi32(xmm0, xmm0);
            _mm_storel_epi64((__m128i*)(c + ldc * j + i), xmm0);
            cm += 8;
        }
    }
    if (i < blk_n)
    {
        for (j = 0; j < blk_m; j++)
        {
            c[ldc * j + i] = cm[0] + cm[1] + cm[2] + cm[3];
            cm += 4;
        }
    }
}

void chgemm_kernel_sse4_u8s8_k64(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc)
{
    __m128i xmm0, xmm1, xmm2, xmm3;
    __m128i xmm4, xmm5, xmm6, xmm7;
    __m128i xmm8, xmm9, xmm10, xmm11;
    __m128i xmm12, xmm13;

    int i, j;
    unsigned char *ap;

    for (i = 0; i <= n - 2; i += 2)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm1 = _mm_load_si128((const __m128i*)(b_loc + 16));
        xmm2 = _mm_load_si128((const __m128i*)(b_loc + 32));
        xmm3 = _mm_load_si128((const __m128i*)(b_loc + 48));
        xmm4 = _mm_load_si128((const __m128i*)(b_loc + 64));
        xmm5 = _mm_load_si128((const __m128i*)(b_loc + 80));
        xmm6 = _mm_load_si128((const __m128i*)(b_loc + 96));
        xmm7 = _mm_load_si128((const __m128i*)(b_loc + 112));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)(c_loc + 0));
            xmm9 = _mm_load_si128((const __m128i*)(c_loc + 4));

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm4, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 16));

            xmm11 = _mm_maddubs_epi16(xmm1, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm5, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 32));

            xmm11 = _mm_maddubs_epi16(xmm2, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm6, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 48));

            xmm11 = _mm_maddubs_epi16(xmm3, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm7, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            _mm_store_si128((__m128i*)(c_loc + 0), xmm8);
            _mm_store_si128((__m128i*)(c_loc + 4), xmm9);

            c_loc += 8;
            ap += 64;
        }
        b_loc += 2 * 64;
    }
    if (i < n)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm1 = _mm_load_si128((const __m128i*)(b_loc + 16));
        xmm2 = _mm_load_si128((const __m128i*)(b_loc + 32));
        xmm3 = _mm_load_si128((const __m128i*)(b_loc + 48));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)c_loc);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 16));

            xmm11 = _mm_maddubs_epi16(xmm1, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 32));

            xmm11 = _mm_maddubs_epi16(xmm2, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 48));

            xmm11 = _mm_maddubs_epi16(xmm3, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            _mm_store_si128((__m128i*)c_loc, xmm8);

            c_loc += 4;
            ap += 64;
        }
        b_loc += 64;
    }
}

void chgemm_kernel_sse4_u8s8_k48(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc)
{
    __m128i xmm0, xmm1, xmm2, xmm3;
    __m128i xmm4, xmm5, xmm6, xmm7;
    __m128i xmm8, xmm9, xmm10, xmm11;
    __m128i xmm12, xmm13;

    int i, j;
    unsigned char *ap;

    for (i = 0; i <= n - 2; i += 2)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm1 = _mm_load_si128((const __m128i*)(b_loc + 16));
        xmm2 = _mm_load_si128((const __m128i*)(b_loc + 32));
        xmm4 = _mm_load_si128((const __m128i*)(b_loc + 48));
        xmm5 = _mm_load_si128((const __m128i*)(b_loc + 64));
        xmm6 = _mm_load_si128((const __m128i*)(b_loc + 80));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)(c_loc + 0));
            xmm9 = _mm_load_si128((const __m128i*)(c_loc + 4));

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm4, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 16));

            xmm11 = _mm_maddubs_epi16(xmm1, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm5, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 32));

            xmm11 = _mm_maddubs_epi16(xmm2, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm6, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            _mm_store_si128((__m128i*)(c_loc + 0), xmm8);
            _mm_store_si128((__m128i*)(c_loc + 4), xmm9);

            c_loc += 8;
            ap += 48;
        }
        b_loc += 2 * 48;
    }
    if (i < n)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm1 = _mm_load_si128((const __m128i*)(b_loc + 16));
        xmm2 = _mm_load_si128((const __m128i*)(b_loc + 32));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)c_loc);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 16));

            xmm11 = _mm_maddubs_epi16(xmm1, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 32));

            xmm11 = _mm_maddubs_epi16(xmm2, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            _mm_store_si128((__m128i*)c_loc, xmm8);

            c_loc += 4;
            ap += 48;
        }
        b_loc += 48;
    }
}

void chgemm_kernel_sse4_u8s8_k32(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc)
{
    __m128i xmm0, xmm1, xmm2, xmm3;
    __m128i xmm4, xmm5, xmm6, xmm7;
    __m128i xmm8, xmm9, xmm10, xmm11;
    __m128i xmm12, xmm13;

    int i, j;
    unsigned char *ap;

    for (i = 0; i <= n - 2; i += 2)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm1 = _mm_load_si128((const __m128i*)(b_loc + 16));
        xmm4 = _mm_load_si128((const __m128i*)(b_loc + 32));
        xmm5 = _mm_load_si128((const __m128i*)(b_loc + 48));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)(c_loc + 0));
            xmm9 = _mm_load_si128((const __m128i*)(c_loc + 4));

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm4, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 16));

            xmm11 = _mm_maddubs_epi16(xmm1, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm5, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            _mm_store_si128((__m128i*)(c_loc + 0), xmm8);
            _mm_store_si128((__m128i*)(c_loc + 4), xmm9);

            c_loc += 8;
            ap += 32;
        }
        b_loc += 2 * 32;
    }
    if (i < n)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm1 = _mm_load_si128((const __m128i*)(b_loc + 16));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)c_loc);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 16));

            xmm11 = _mm_maddubs_epi16(xmm1, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            _mm_store_si128((__m128i*)c_loc, xmm8);

            c_loc += 4;
            ap += 32;
        }
        b_loc += 32;
    }
}

void chgemm_kernel_sse4_u8s8_k16(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc)
{
    __m128i xmm0, xmm1, xmm2, xmm3;
    __m128i xmm4, xmm5, xmm6, xmm7;
    __m128i xmm8, xmm9, xmm10, xmm11;
    __m128i xmm12, xmm13;

    int i, j;
    unsigned char *ap;

    for (i = 0; i <= n - 2; i += 2)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));
        xmm4 = _mm_load_si128((const __m128i*)(b_loc + 16));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)(c_loc + 0));
            xmm9 = _mm_load_si128((const __m128i*)(c_loc + 4));

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm12 = _mm_maddubs_epi16(xmm4, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm13 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);
            xmm12 = _mm_shuffle_epi32(xmm12, 0x4e);
            xmm12 = _mm_cvtepi16_epi32(xmm12);
            xmm9 = _mm_add_epi32(xmm9, xmm12);

            _mm_store_si128((__m128i*)(c_loc + 0), xmm8);
            _mm_store_si128((__m128i*)(c_loc + 4), xmm9);

            c_loc += 8;
            ap += 16;
        }
        b_loc += 2 * 16;
    }
    if (i < n)
    {
        ap = a_loc;

        xmm0 = _mm_load_si128((const __m128i*)(b_loc + 0));

        for (j = 0; j < m; j++)
        {
            xmm8 = _mm_load_si128((const __m128i*)c_loc);

            xmm10 = _mm_load_si128((const __m128i*)(ap + 0));

            xmm11 = _mm_maddubs_epi16(xmm0, xmm10);
            xmm13 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm13);
            xmm11 = _mm_shuffle_epi32(xmm11, 0x4e);
            xmm11 = _mm_cvtepi16_epi32(xmm11);
            xmm8 = _mm_add_epi32(xmm8, xmm11);

            _mm_store_si128((__m128i*)c_loc, xmm8);

            c_loc += 4;
            ap += 16;
        }
        b_loc += 16;
    }
}

