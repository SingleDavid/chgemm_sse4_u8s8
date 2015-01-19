#ifndef _CHGEMM_KERNEL_H
#define _CHGEMM_KERNEL_H

#include <stdlib.h>

void chgemm_build_loc_sse4_u8s8(void *src, void *dst, int lds, int block_row, int block_col);
void chgemm_c_loc_clear_sse4_u8s8(void *c, size_t size);
void chgemm_c_back_sse4_u8s8(int *cm, int *c, int ldc, int blk_m, int blk_n);
void chgemm_kernel_sse4_u8s8_k64(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc);
void chgemm_kernel_sse4_u8s8_k48(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc);
void chgemm_kernel_sse4_u8s8_k32(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc);
void chgemm_kernel_sse4_u8s8_k16(unsigned char *a_loc, int m, char *b_loc, int n, int *c_loc);

#endif

