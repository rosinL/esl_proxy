#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "mem_pool.h"
#include "ring_buf.h"

#include "qwen3_dynamic_tensormap.h"
#define MEM_POOL_BYTES (2048UL * 1024UL * 1024UL)

static uint8_t *g_mem_pool_storage;

static int cmp_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    return (va > vb) - (va < vb);
}

static void dump_dag_orig(uint32_t total_tasks) {
    for (uint32_t tid = 0; tid < total_tasks; tid++) {
        struct task_desc *t = &g_basic_buf[tid & RING_MASK];
        printf("TASK %u: type=%u mode=%u count=%u dur=%u tc=%u sc=%u",
               tid, t->type, t->mode, t->count, t->duration,
               t->tensor_cnt, t->scalar_cnt);
        printf(" data=[");
        for (uint32_t i = 0; i < t->tensor_cnt; i++)
            printf("%lu%s", (unsigned long)t->data[i],
                   i < t->tensor_cnt - 1 ? "," : "");
        printf("] scalar=[");
        for (uint32_t i = 0; i < t->scalar_cnt; i++)
            printf("%ld%s", (long)t->scalar[i],
                   i < t->scalar_cnt - 1 ? "," : "");
        printf("]\n");

        struct predecessor_list *p = &g_predecessors[tid];
        if (p->cnt > 0 && p->exp != NULL) {
            uint32_t *sorted = malloc((size_t)p->cnt * sizeof(uint32_t));
            if (sorted) {
                memcpy(sorted, p->exp, (size_t)p->cnt * sizeof(uint32_t));
                qsort(sorted, (size_t)p->cnt, sizeof(uint32_t), cmp_u32);
                printf("PRED %u: cnt=%d [", tid, p->cnt);
                for (int i = 0; i < p->cnt; i++)
                    printf("%u%s", sorted[i], i < p->cnt - 1 ? "," : "");
                printf("]\n");
                free(sorted);
            }
        } else {
            printf("PRED %u: cnt=0 []\n", tid);
        }
    }
}

int main(void) {
    g_mem_pool_storage = malloc(MEM_POOL_BYTES);
    if (!g_mem_pool_storage) { fprintf(stderr, "malloc failed for mem_pool\n"); return 1; }
    mem_pool_init(&g_mem_pool, g_mem_pool_storage, MEM_POOL_BYTES);
    ring_buf_init();
    init_predecessors();

    aicpu_orchestration_entry(0);

    dump_dag_orig((uint32_t)atomic_load(&g_task_id));
    return 0;
}
