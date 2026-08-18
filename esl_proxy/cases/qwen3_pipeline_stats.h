#ifndef QWEN3_PIPELINE_STATS_H
#define QWEN3_PIPELINE_STATS_H

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include "ring_buf.h"

typedef struct {
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t task_count;
} ThreadStats;

typedef struct {
    ThreadStats s;
    char _pad[64 - sizeof(ThreadStats)];
} AlignedStats;

#ifndef NUM_ORCH_TILES
#define NUM_ORCH_TILES 6
#endif

static AlignedStats g_alloc_stat;
static AlignedStats g_worker_stats[NUM_ORCH_TILES];

static uint64_t g_pipeline_start_ns;
static uint64_t g_pipeline_end_ns;

static int g_num_workers = NUM_ORCH_TILES;

static inline void print_pipeline_stats(uint32_t total_tasks, uint32_t total_subtasks) {
    uint64_t elapsed_ns = g_pipeline_end_ns - g_pipeline_start_ns;
    double throughput = (double)total_tasks * 1000.0 / (double)elapsed_ns;
    double sub_tp = (double)total_subtasks * 1000.0 / (double)elapsed_ns;

    printf("==== Pipeline Statistics ====\n");
    printf("pipeline_elapsed_ns:  %llu\n", (unsigned long long)elapsed_ns);
    printf("total_tasks:          %u\n", total_tasks);
    printf("total_subtasks:       %u\n", total_subtasks);
    printf("num_workers:          %d\n", g_num_workers);
    printf("throughput:           %.2f MTasks/s\n", throughput);
    printf("subtask_throughput:   %.2f MTasks/s\n", sub_tp);
    printf("\n");

    printf("---- Per-Thread ----\n");
    printf("thread          role     tasks   elapsed_ns  throughput(MTasks/s)\n");

    uint64_t a_elapsed = g_alloc_stat.s.end_ns - g_alloc_stat.s.start_ns;
    double a_tp = (a_elapsed > 0)
        ? (double)g_alloc_stat.s.task_count * 1000.0 / (double)a_elapsed : 0.0;
    printf("main            alloc    %4u    %10llu    %.2f\n",
           g_alloc_stat.s.task_count,
           (unsigned long long)a_elapsed, a_tp);

    uint64_t worker_wall = 0;
    for (int w = 0; w < g_num_workers; w++) {
        uint64_t w_el = g_worker_stats[w].s.end_ns - g_worker_stats[w].s.start_ns;
        double w_tp = (w_el > 0)
            ? (double)g_worker_stats[w].s.task_count * 1000.0 / (double)w_el : 0.0;
        printf("worker-%d        worker   %4u    %10llu    %.2f\n",
               w, g_worker_stats[w].s.task_count,
               (unsigned long long)w_el, w_tp);
        if (w_el > worker_wall) worker_wall = w_el;
    }

    printf("\n---- Pipeline Efficiency ----\n");
    printf("alloc   wall: %llu ns (main thread, pre-pipeline)\n", (unsigned long long)a_elapsed);
    if (worker_wall > 0) {
        printf("worker  wall: %llu ns (%d threads)\n",
               (unsigned long long)worker_wall, g_num_workers);
    }
    printf("overall wall: %llu ns, throughput=%.2f MTasks/s\n",
           (unsigned long long)elapsed_ns, throughput);
}

#ifdef DAG_DUMP
static int cmp_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    return (va > vb) - (va < vb);
}

static inline void dump_dag(uint32_t total_tasks) {
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
            } else {
                printf("PRED %u: cnt=%d [malloc failed]\n", tid, p->cnt);
            }
        } else {
            printf("PRED %u: cnt=0 []\n", tid);
        }
    }
}
#endif /* DAG_DUMP */

#endif /* QWEN3_PIPELINE_STATS_H */
