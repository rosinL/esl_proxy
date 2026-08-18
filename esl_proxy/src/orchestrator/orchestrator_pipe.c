#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.h"
#include "mem_pool.h"
#include "ring_buf.h"

extern void init_predecessors(void);

#include "qwen3_pipeline.h"

#define MEM_POOL_BYTES (2048UL * 1024UL * 1024UL)

static uint8_t *g_mem_pool_storage;

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        int n = atoi(argv[1]);
        if (n < 1 || n > NUM_ORCH_TILES) {
            fprintf(stderr, "Usage: %s [num_workers 1..%d]  (default %d)\n",
                    argv[0], NUM_ORCH_TILES, NUM_ORCH_TILES);
            return 1;
        }
        g_num_workers = n;
    }

    g_mem_pool_storage = malloc(MEM_POOL_BYTES);
    if (!g_mem_pool_storage) { fprintf(stderr, "malloc failed for mem_pool\n"); return 1; }
    mem_pool_init(&g_mem_pool, g_mem_pool_storage, MEM_POOL_BYTES);
    ring_buf_init();
    init_predecessors();

    aicpu_orchestration_entry(0);
    return 0;
}
