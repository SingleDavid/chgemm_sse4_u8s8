#ifndef _CHGEMM_H
#define _CHGEMM_H

#include <stdlib.h>

void *chgemm_alloc(size_t size);
void chgemm_free(void *mem);
void chgemm(int m, int n, int k, unsigned char *a, char *b, int *c);

#endif

