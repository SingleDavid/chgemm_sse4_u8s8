#define _GNU_SOURCE
#include "chgemm.h"
// #include "chgemm_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/mman.h>

#define LOOP_TIME 1024

void thread_bind(int cpu)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    if (pthread_setaffinity_np(pthread_self(),
            sizeof(cpu_set_t), &cpu_set) != 0)
    {
        fprintf(stderr, "Error: cpu[%d] bind failed.\n", cpu);
        exit(0);
    }
}

void save_file(const char *file_name, int *c, int m, int n)
{
    FILE *fp = fopen(file_name, "w");
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fp, "%d ", c[i * n + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s M N K\n", argv[0]);
        exit(0);
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    struct timeval start, end;
    int i, j, k;

    thread_bind(0);

    unsigned char *a = (unsigned char*)chgemm_alloc(M * K * sizeof(unsigned char));
    char *b = (char*)chgemm_alloc(N * K * sizeof(char));
    int *c = (int*)chgemm_alloc(M * N * sizeof(int));

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < K; j++)
        {
            a[i * K + j] = (unsigned char)1;
        }
    }
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < K; j++)
        {
            b[i * K + j] = (char)1;
        }
    }

    // warm up
    chgemm(M, N, K, a, b, c);

    gettimeofday(&start, NULL);
    for (i = 0; i < LOOP_TIME; i++)
    {
        chgemm(M, N, K, a, b, c);
    }
    gettimeofday(&end, NULL);
    printf("time cost %lfs.\n", (end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1e-6) / LOOP_TIME);
    save_file("result.txt", c, M, N);

    chgemm_free(a);
    chgemm_free(b);
    chgemm_free(c);

    return 0;
}

