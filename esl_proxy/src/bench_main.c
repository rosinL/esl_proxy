#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conf.h"
#include "log.h"
#include "mem_pool.h"
#include "ring_buf.h"

#ifndef ORCH_CASE
#define ORCH_CASE qwen3_dynamic_tensormap.h
#endif

#define STR(x) #x
#define XSTR(x) STR(x)

#include XSTR(ORCH_CASE)

void aicpu_orchestration_entry(const uint64_t orch_args);

#define MEM_POOL_BYTES (1024UL * 1024UL * 1024UL)
#define WHEN2FREE_CAP 4096

static uint8_t g_mem_pool_storage[MEM_POOL_BYTES];
static when2free_entry_t g_when2free_entries[WHEN2FREE_CAP];

extern atomic_bool g_orch_is_done;
extern atomic_int g_task_id;
extern int g_subtask_cnt;

int main(void)
{
    mem_pool_init(&g_mem_pool, g_mem_pool_storage, sizeof g_mem_pool_storage);
    mem_pool_init_fifo(&g_mem_pool, g_when2free_entries, WHEN2FREE_CAP);
    ring_buf_init();
    init_predecessors();

    uint64_t start_ns = get_time_ns();
    aicpu_orchestration_entry(0);
    uint64_t end_ns = get_time_ns();
    uint64_t elapsed_ns = end_ns - start_ns;

    int task_cnt = atomic_load(&g_task_id);

    printf("case: %s\n", XSTR(ORCH_CASE));
    printf("task_cnt=%d  elapsed=%llu ns  throughput=%.2f MTasks/s\n",
           task_cnt,
           (unsigned long long)elapsed_ns,
           elapsed_ns > 0 ? (double)task_cnt * 1000.0 / (double)elapsed_ns : 0.0);

    return 0;
}
