#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.h"
#include "orch_config.h"

/* Declared in orc_alloc.c / orc_desc.c (separate translation units to avoid
 * symbol conflicts between qwen3_14b_decoder_alloc.h and
 * qwen3_14b_decoder_desc.h) */
void orc_alloc_call(uint64_t orch_args);
int  orc_desc_call(uint64_t orch_args, int thread_id, int *created_cnt);
void orc_submit_init(int total_tasks);
void orc_submit_task(uint32_t task_id);
extern void init_predecessors(void);

struct desc_thread_arg {
    uint64_t orch_args;
    int thread_id;
    int task_count;
    int created_cnt;
    uint64_t elapsed_ns;
};

struct submit_work_arg {
    int start;
    int end;
    uint64_t elapsed_ns;
};

int desc_thread_count = DESC_THREAD_COUNT;
int submit_thread_count = DESC_THREAD_COUNT;
int desc_batch_size = 64;

static void *alloc_thread_func(void *arg)
{
    uint64_t orch_args = (uint64_t)(uintptr_t)arg;
    orc_alloc_call(orch_args);
    return NULL;
}

static void *desc_thread_func(void *arg)
{
    struct desc_thread_arg *targ = (struct desc_thread_arg *)arg;
    uint64_t t0 = get_time_ns();
    targ->task_count = orc_desc_call(targ->orch_args, targ->thread_id, &targ->created_cnt);
    uint64_t t1 = get_time_ns();
    targ->elapsed_ns = t1 - t0;
    return NULL;
}

static void *submit_thread_func(void *arg)
{
    struct submit_work_arg *w = (struct submit_work_arg *)arg;
    uint64_t t0 = get_time_ns();
    for (int t = w->start; t < w->end; t++)
        orc_submit_task((uint32_t)t);
    w->elapsed_ns = get_time_ns() - t0;
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc >= 2) {
        desc_thread_count = atoi(argv[1]);
        if (desc_thread_count <= 0) {
            fprintf(stderr, "Usage: %s [desc_thread_count] [submit_thread_count]\n", argv[0]);
            return 1;
        }
    }
    if (argc >= 3) {
        submit_thread_count = atoi(argv[2]);
        if (submit_thread_count <= 0) {
            fprintf(stderr, "Usage: %s [desc_thread_count] [submit_thread_count]\n", argv[0]);
            return 1;
        }
    }

    pthread_t alloc_thread;
    pthread_t *desc_threads = malloc((size_t)desc_thread_count * sizeof(pthread_t));
    if (!desc_threads) {
        fprintf(stderr, "Failed to allocate thread array\n");
        return 1;
    }

    uint64_t start_ns = get_time_ns();

    /* Phase 1: alloc (1 thread, serial) */
    pthread_create(&alloc_thread, NULL, alloc_thread_func, (void *)(uintptr_t)0);
    pthread_join(alloc_thread, NULL);

    /* Phase 2: desc (N threads, parallel) */
    struct desc_thread_arg *desc_args = malloc((size_t)desc_thread_count * sizeof(struct desc_thread_arg));
    if (!desc_args) {
        fprintf(stderr, "Failed to allocate desc thread args\n");
        free(desc_threads);
        return 1;
    }
    for (int i = 0; i < desc_thread_count; i++) {
        desc_args[i].orch_args = 0;
        desc_args[i].thread_id = i;
        pthread_create(&desc_threads[i], NULL, desc_thread_func, &desc_args[i]);
    }
    for (int i = 0; i < desc_thread_count; i++)
        pthread_join(desc_threads[i], NULL);

    printf("desc_thread throughput (MTasks/s):\n");
    int total_cnt = 0;
    for (int i = 0; i < desc_thread_count; i++) {
        double throughput = (double)desc_args[i].task_count / (double)desc_args[i].elapsed_ns * 1000.0;
        printf("  thread %2d: tasks=%d  created=%d  time=%llu ns  throughput=%.2f MTasks/s\n",
               desc_args[i].thread_id, desc_args[i].task_count, desc_args[i].created_cnt,
               (unsigned long long)desc_args[i].elapsed_ns, throughput);
        total_cnt += desc_args[i].created_cnt;
    }

    int total_tasks = desc_args[0].task_count;
    free(desc_args);
    free(desc_threads);

    /* Phase 3: submit (parallel, direct Tensor overlap) */
    orc_submit_init(total_tasks);
    init_predecessors();

    int M = submit_thread_count;
    pthread_t *sub_threads = malloc((size_t)M * sizeof(pthread_t));
    struct submit_work_arg *sub_args = malloc((size_t)M * sizeof(struct submit_work_arg));

    uint64_t t3 = get_time_ns();
    for (int i = 0; i < M; i++) {
        sub_args[i].start = (total_tasks * i) / M;
        sub_args[i].end   = (total_tasks * (i + 1)) / M;
        pthread_create(&sub_threads[i], NULL, submit_thread_func, &sub_args[i]);
    }
    for (int i = 0; i < M; i++)
        pthread_join(sub_threads[i], NULL);
    uint64_t t3_end = get_time_ns();

    printf("submit (%d threads): %llu ns\n",
           M, (unsigned long long)(t3_end - t3));

    uint64_t end_ns = get_time_ns();
    printf("orchestrator total elapsed (1 alloc + %d desc + %d submit): %llu ns\n",
           desc_thread_count, M, (unsigned long long)(end_ns - start_ns));
    printf("desc=%d  submit=%d\n", total_cnt, total_tasks);

    free(sub_args);
    free(sub_threads);
    return 0;
}
