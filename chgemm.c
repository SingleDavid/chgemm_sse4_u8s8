#include "chgemm_kernel.h"
#include "chgemm.h"

#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOC_SIZE (64*64*2+64*64*16)

static __thread void *s_buf = NULL;

void *chgemm_alloc(size_t size)
{
    return _mm_malloc(size, 64);
}

void chgemm_free(void *mem)
{
    _mm_free(mem);
}

static void chgemm_init()
{
    if (s_buf == NULL)
    {
        s_buf = chgemm_alloc(LOC_SIZE);
    }
}

void chgemm(int m, int n, int k, unsigned char *a, char *b, int *c)
{
    if (k & 0xF)
    {
        fprintf(stderr, "Error: k must a multiple of 16.\n");
        exit(0);
    }
    chgemm_init();

    int i, j, kk;
    int block_row, block_col;
    int lda = k;
    int ldb = k;
    int ldc = n;

    unsigned char* ap = a;
    char *bp;
    int *cp = c;

    void *buf = s_buf;
    unsigned char *a_loc = (unsigned char*)buf;
    char *b_loc = (char*)(buf + 64 * 64);
    int *c_loc = (int*)(buf + 64 * 64 * 2);

    for (i = 0; i <= m - 64; i += 64)
    {
        bp = b;
        for (j = 0; j <= n - 64; j += 64)
        {
            chgemm_c_loc_clear_sse4_u8s8(c_loc, 64 * 64 * 16);

            // m = n = k = 64
            for (kk = 0; kk <= k - 64; kk += 64)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, 64, 64);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, 64, 64);
                chgemm_kernel_sse4_u8s8_k64(a_loc, 64, b_loc, 64, c_loc);
            }

            // m = n = 64
            if (k - kk)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, 64, k - kk);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, 64, k - kk);
                if (k - kk >= 48)
                {
                    chgemm_kernel_sse4_u8s8_k48(a_loc, 64, b_loc, 64, c_loc);
                }
                else if (k - kk >= 32)
                {
                    chgemm_kernel_sse4_u8s8_k32(a_loc, 64, b_loc, 64, c_loc);
                }
                else if (k - kk >= 16)
                {
                    chgemm_kernel_sse4_u8s8_k16(a_loc, 64, b_loc, 64, c_loc);
                }
            }
            chgemm_c_back_sse4_u8s8(c_loc, cp + j, ldc, 64, 64);
            bp += 64 * ldb;
        }
        if (n - j)
        {
            chgemm_c_loc_clear_sse4_u8s8(c_loc, 64 * (n - j) * 16);

            // m = k = 64
            for (kk = 0; kk <= k - 64; kk += 64)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, 64, 64);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, n - j, 64);
                chgemm_kernel_sse4_u8s8_k64(a_loc, 64, b_loc, n - j, c_loc);
            }

            // m = 64
            if (k - kk)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, 64, k - kk);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, n - j, k - kk);
                if (k - kk >= 48)
                {
                    chgemm_kernel_sse4_u8s8_k48(a_loc, 64, b_loc, n - j, c_loc);
                }
                else if (k - kk >= 32)
                {
                    chgemm_kernel_sse4_u8s8_k32(a_loc, 64, b_loc, n - j, c_loc);
                }
                else if (k - kk >= 16)
                {
                    chgemm_kernel_sse4_u8s8_k16(a_loc, 64, b_loc, n - j, c_loc);
                }
            }
            chgemm_c_back_sse4_u8s8(c_loc, cp + j, ldc, 64, n - j);
        }
        ap += 64 * lda;
        cp += 64 * ldc;
    }
    if (m - i)
    {
        bp = b;
        for (j = 0; j <= n - 64; j += 64)
        {
            chgemm_c_loc_clear_sse4_u8s8(c_loc, (m - i) * 64 * 16);

            // n = k = 64
            for (kk = 0; kk <= k - 64; kk += 64)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, m - i, 64);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, 64, 64);
                chgemm_kernel_sse4_u8s8_k64(a_loc, m - i, b_loc, 64, c_loc);
            }

            // n = 64
            if (k - kk)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, m - i, k - kk);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, 64, k - kk);
                if (k - kk >= 48)
                {
                    chgemm_kernel_sse4_u8s8_k48(a_loc, m - i, b_loc, 64, c_loc);
                }
                else if (k - kk >= 32)
                {
                    chgemm_kernel_sse4_u8s8_k32(a_loc, m - i, b_loc, 64, c_loc);
                }
                else if (k - kk >= 16)
                {
                    chgemm_kernel_sse4_u8s8_k16(a_loc, m - i, b_loc, 64, c_loc);
                }
            }
            chgemm_c_back_sse4_u8s8(c_loc, cp + j, ldc, m - i, 64);
            bp += 64 * ldb;
        }
        if (n - j)
        {
            chgemm_c_loc_clear_sse4_u8s8(c_loc, (m - i) * (n - j) * 16);

            // k = 64
            for (kk = 0; kk <= k - 64; kk += 64)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, m - i, 64);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, n - j, 64);
                chgemm_kernel_sse4_u8s8_k64(a_loc, m - i, b_loc, n - j, c_loc);
            }

            // no one is 64
            if (k - kk)
            {
                chgemm_build_loc_sse4_u8s8(ap + kk, a_loc, lda, m - i, k - kk);
                chgemm_build_loc_sse4_u8s8(bp + kk, b_loc, ldb, n - j, k - kk);
                if (k - kk >= 48)
                {
                    chgemm_kernel_sse4_u8s8_k48(a_loc, m - i, b_loc, n - j, c_loc);
                }
                else if (k - kk >= 32)
                {
                    chgemm_kernel_sse4_u8s8_k32(a_loc, m - i, b_loc, n - j, c_loc);
                }
                else if (k - kk >= 16)
                {
                    chgemm_kernel_sse4_u8s8_k16(a_loc, m - i, b_loc, n - j, c_loc);
                }
            }
            chgemm_c_back_sse4_u8s8(c_loc, cp + j, ldc, m - i, n - j);
        }
    }
}

