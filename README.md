chgemm_sse4_u8s8:
a sse4 version of integer gemm.

interface:
void chgemm(int m, int n, int k, unsigned char *a, char *b, int *c);

note:
m, n can be any integer larger than 0, but k must be a multiple of 16.
