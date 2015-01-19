gcc -Ofast -msse4 -c chgemm_kernel_sse4_u8s8.c
gcc -Ofast -msse4 -c chgemm.c
gcc -pthread -Ofast -c main.c
gcc -pthread -Ofast -o chgemm main.o chgemm_kernel_sse4_u8s8.o chgemm.o
