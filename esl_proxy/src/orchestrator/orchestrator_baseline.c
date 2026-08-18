#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.h"
#include "mem_pool.h"
#include "ring_buf.h"

extern void init_predecessors(void);

#include "qwen3_dynamic_tensormap.h"

#define MEM_POOL_BYTES (2048UL * 1024UL * 1024UL)
static uint8_t *g_mem_pool_storage;

int main(void) {
    g_mem_pool_storage = malloc(MEM_POOL_BYTES);
    if (!g_mem_pool_storage) { fprintf(stderr, "malloc failed\n"); return 1; }
    mem_pool_init(&g_mem_pool, g_mem_pool_storage, MEM_POOL_BYTES);
    ring_buf_init();
    init_predecessors();

    uint64_t start_ns = get_time_ns();
    aicpu_orchestration_entry(0);
    uint64_t end_ns = get_time_ns();
    uint64_t elapsed_ns = end_ns - start_ns;

    uint32_t total_tasks = (uint32_t)atomic_load(&g_task_id);
    double throughput = (double)total_tasks * 1000.0 / (double)elapsed_ns;

    printf("==== Baseline (Single-Thread) Statistics ====\n");
    printf("elapsed_ns:           %llu\n", (unsigned long long)elapsed_ns);
    printf("total_tasks:          %u\n", total_tasks);
    printf("total_subtasks:       %d\n", g_subtask_cnt);
    printf("throughput:           %.2f MTasks/s\n", throughput);
    return 0;
}
