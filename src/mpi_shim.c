// fitxategi honek QOA-ren eta MPI liburutegiaren arteko geruza mehea da (`mpi` ezaugarria, src/mpi.rs).
// mpicc-ekin konpilatzen da, beraz instalatutako MPI-ren mpi.h-rekin (openmpi, mpich, cray-mpich...):
// rust-ek ez du MPI-ren ABI-a ezagutu behar. mezu handiak 1 GiB-eko zatietan doaz (MPI-ren kontagailuak int dira).

#include <mpi.h>
#include <stdint.h>

#define CHUNK ((uint64_t)1 << 30)

int qoa_mpi_init(void) {
    int done = 0;
    MPI_Initialized(&done);
    if (done) {
        return 0;
    }
    int provided = 0;
    return MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS ? 0 : 1;
}

void qoa_mpi_finalize(void) {
    int done = 0;
    MPI_Finalized(&done);
    if (!done) {
        MPI_Finalize();
    }
}

int qoa_mpi_rank(void) {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int qoa_mpi_size(void) {
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

void qoa_mpi_barrier(void) {
    MPI_Barrier(MPI_COMM_WORLD);
}

// luzera (1. etiketa) eta gero edukia zatika (2. etiketa)
void qoa_mpi_send(const uint8_t *p, uint64_t n, int to) {
    MPI_Send(&n, 1, MPI_UINT64_T, to, 1, MPI_COMM_WORLD);
    for (uint64_t off = 0; off < n; off += CHUNK) {
        uint64_t k = n - off < CHUNK ? n - off : CHUNK;
        MPI_Send(p + off, (int)k, MPI_BYTE, to, 2, MPI_COMM_WORLD);
    }
}

uint64_t qoa_mpi_recv_len(int from) {
    uint64_t n = 0;
    MPI_Recv(&n, 1, MPI_UINT64_T, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return n;
}

void qoa_mpi_recv_data(uint8_t *p, uint64_t n, int from) {
    for (uint64_t off = 0; off < n; off += CHUNK) {
        uint64_t k = n - off < CHUNK ? n - off : CHUNK;
        MPI_Recv(p + off, (int)k, MPI_BYTE, from, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// tamaina bereko bi buffer trukatu bikote batekin (3. etiketa)
void qoa_mpi_exchange(const uint8_t *s, uint8_t *r, uint64_t n, int partner) {
    for (uint64_t off = 0; off < n; off += CHUNK) {
        uint64_t k = n - off < CHUNK ? n - off : CHUNK;
        MPI_Sendrecv(s + off, (int)k, MPI_BYTE, partner, 3, r + off, (int)k, MPI_BYTE, partner, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
