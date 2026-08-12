#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200112L

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "orch_config.h"
#include "ring_buf.h"

extern struct predecessor_list g_predecessors[];

static int dump_predecessors(const char *path, int total_task_cnt)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        return -1;
    }
    fprintf(fp, "# predecessors dump total_task_cnt=%d\n", total_task_cnt);
    for (int tid = 0; tid < total_task_cnt; tid++) {
        struct predecessor_list *pl = &g_predecessors[tid];
        uint32_t cnt = pl->cnt;
        uint32_t tmp[256];
        if (cnt > 256)
            cnt = 256;
        for (uint32_t i = 0; i < cnt; i++)
            tmp[i] = pl->exp[i];
        for (uint32_t i = 1; i < cnt; i++) {
            uint32_t v = tmp[i];
            uint32_t j = i;
            while (j > 0 && tmp[j - 1] > v) {
                tmp[j] = tmp[j - 1];
                j--;
            }
            tmp[j] = v;
        }
        fprintf(fp, "%d %u", tid, cnt);
        for (uint32_t i = 0; i < cnt; i++)
            fprintf(fp, " %u", tmp[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}

/* Declared in orc_alloc.c / orc_desc.c (separate translation units to avoid
 * symbol conflicts between qwen3_14b_decoder_alloc.h and
 * qwen3_14b_decoder_desc.h) */
void orc_alloc_call(uint64_t orch_args);
int orc_desc_call(uint64_t orch_args, int thread_id, int *created_cnt);
void orc_submit_init(int thread_count);
void orc_submit_cleanup(void);
int orc_submit_call(int thread_id, int total_tasks, int *submit_cnt);

struct desc_thread_arg {
    uint64_t orch_args;
    int thread_id;
    int task_count;
    int created_cnt;
    uint64_t elapsed_ns;
    int core_id;
};

struct submit_thread_arg {
    int thread_id;
    int total_tasks;
    int submit_cnt;
    uint64_t elapsed_ns;
    int core_id;
};

int desc_thread_count = DESC_THREAD_COUNT;
int desc_batch_size = 32;

static pthread_barrier_t g_phase_barrier;

static void pin_cpu(int core_id)
{
    if (core_id < 0)
        return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static void *alloc_thread_func(void *arg)
{
    uint64_t orch_args = (uint64_t)(uintptr_t)arg;
    orc_alloc_call(orch_args);
    return NULL;
}

static void *desc_thread_func(void *arg)
{
    struct desc_thread_arg *targ = (struct desc_thread_arg *)arg;
    pin_cpu(targ->core_id);
    pthread_barrier_wait(&g_phase_barrier);
    uint64_t t0 = get_time_ns();
    targ->task_count = orc_desc_call(targ->orch_args, targ->thread_id, &targ->created_cnt);
    uint64_t t1 = get_time_ns();
    targ->elapsed_ns = t1 - t0;
    return NULL;
}

static void *submit_thread_func(void *arg)
{
    struct submit_thread_arg *targ = (struct submit_thread_arg *)arg;
    pin_cpu(targ->core_id);
    pthread_barrier_wait(&g_phase_barrier);
    uint64_t t0 = get_time_ns();
    orc_submit_call(targ->thread_id, targ->total_tasks, &targ->submit_cnt);
    uint64_t t1 = get_time_ns();
    targ->elapsed_ns = t1 - t0;
    return NULL;
}

static void parse_cpu_range(const char *spec, int *out, int max_count)
{
    for (int i = 0; i < max_count; i++)
        out[i] = -1;
    if (!spec || !spec[0])
        return;

    int idx = 0;
    const char *p = spec;
    while (*p && idx < max_count) {
        int lo = 0, hi = 0;
        int n = 0;
        if (sscanf(p, "%d-%d%n", &lo, &hi, &n) == 2) {
            for (int c = lo; c <= hi && idx < max_count; c++)
                out[idx++] = c;
        } else if (sscanf(p, "%d%n", &lo, &n) == 1) {
            out[idx++] = lo;
        } else {
            break;
        }
        p += n;
        if (*p == ',')
            p++;
    }
}

int main(int argc, char *argv[])
{
    int desc_cpu[256];
    parse_cpu_range(NULL, desc_cpu, 0);

    /* Usage: orchestrator [thread_count] [cpu_range] */
    if (argc >= 2) {
        desc_thread_count = atoi(argv[1]);
        if (desc_thread_count <= 0) {
            fprintf(stderr,
                "Usage: %s [thread_count] [cpu_range]\n"
                "  cpu_range: e.g. 0-7 or 0,2,4,6 or 0-3,8-11 (shared by desc+submit)\n"
                "  omit cpu_range for no pinning\n",
                argv[0]);
            return 1;
        }
    }
    if (argc >= 3)
        parse_cpu_range(argv[2], desc_cpu, desc_thread_count);

    int has_pin = (argc >= 3);
    printf("threads=%d  cpu_pin=%s\n",
           desc_thread_count, has_pin ? "yes" : "no");

    pthread_t alloc_thread;
    pthread_t *desc_threads = malloc((size_t)desc_thread_count * sizeof(pthread_t));
    if (!desc_threads) {
        fprintf(stderr, "Failed to allocate thread array\n");
        return 1;
    }

    uint64_t start_ns, end_ns, elapsed_ns;
    start_ns = get_time_ns();

    /* 1 orchestrator_alloc thread */
    pthread_create(&alloc_thread, NULL, alloc_thread_func, (void *)(uintptr_t)0);
    pthread_join(alloc_thread, NULL);

    /* N orchestrator_desc threads in parallel (started after alloc thread finishes) */
    struct desc_thread_arg *desc_args = malloc((size_t)desc_thread_count * sizeof(struct desc_thread_arg));
    if (!desc_args) {
        fprintf(stderr, "Failed to allocate desc thread args\n");
        free(desc_threads);
        return 1;
    }
    pthread_barrier_init(&g_phase_barrier, NULL, desc_thread_count);
    for (int i = 0; i < desc_thread_count; i++) {
        desc_args[i].orch_args = 0;
        desc_args[i].thread_id = i;
        desc_args[i].core_id = desc_cpu[i];
        pthread_create(&desc_threads[i], NULL, desc_thread_func, &desc_args[i]);
    }

    for (int i = 0; i < desc_thread_count; i++) {
        pthread_join(desc_threads[i], NULL);
    }

    pthread_barrier_destroy(&g_phase_barrier);

    uint64_t desc_end_ns = get_time_ns();

    /* Print per-thread throughput: desc_task_id / execution_time (MTasks/s) */
    printf("desc_thread throughput (MTasks/s):\n");
    int total_cnt = 0;
    uint64_t desc_max_ns = 0;
    for (int i = 0; i < desc_thread_count; i++) {
        /* throughput = tasks / us   (because MTasks/s = tasks / (us * 1e-6) * 1e-6 = tasks / us) */
        double throughput = (double)desc_args[i].task_count / (double)desc_args[i].elapsed_ns * (double)1000.0;
        double time_240_us = 240.0 / throughput;
        printf("  thread %2d: tasks=%d  created=%d  time=%llu ns  throughput=%.2f MTasks/s  time_240=%.2f us\n",
               desc_args[i].thread_id,
               desc_args[i].task_count,
               desc_args[i].created_cnt,
               (unsigned long long)desc_args[i].elapsed_ns,
               throughput,
               time_240_us);
        total_cnt += desc_args[i].created_cnt;
        if (desc_args[i].elapsed_ns > desc_max_ns)
            desc_max_ns = desc_args[i].elapsed_ns;
    }

    int total_tasks = desc_args[0].task_count;

    /* Phase 3: Parallel tm_submit (dependency discovery)
     * Runs after all desc threads complete, so g_task_tensor_buf is fully
     * populated. Uses two sub-phases with a barrier:
     *   3a: parallel insert all output tensors into tensormap
     *   --- barrier ---
     *   3b: parallel lookup predecessors for all input tensors */
    orc_submit_init(desc_thread_count);

    pthread_t *submit_threads = malloc((size_t)desc_thread_count * sizeof(pthread_t));
    struct submit_thread_arg *submit_args = malloc((size_t)desc_thread_count * sizeof(struct submit_thread_arg));
    if (!submit_threads || !submit_args) {
        fprintf(stderr, "Failed to allocate submit thread arrays\n");
        free(submit_threads);
        free(submit_args);
        free(desc_args);
        free(desc_threads);
        return 1;
    }

    pthread_barrier_init(&g_phase_barrier, NULL, desc_thread_count);

    for (int i = 0; i < desc_thread_count; i++) {
        submit_args[i].thread_id = i;
        submit_args[i].total_tasks = total_tasks;
        submit_args[i].submit_cnt = 0;
        submit_args[i].core_id = desc_cpu[i];
        pthread_create(&submit_threads[i], NULL, submit_thread_func, &submit_args[i]);
    }

    for (int i = 0; i < desc_thread_count; i++) {
        pthread_join(submit_threads[i], NULL);
    }

    pthread_barrier_destroy(&g_phase_barrier);

    orc_submit_cleanup();

    end_ns = get_time_ns();
    elapsed_ns = end_ns - start_ns;

    /* Print submit phase throughput */
    printf("\nsubmit_thread throughput (MTasks/s):\n");
    int total_submit_cnt = 0;
    uint64_t submit_max_ns = 0;
    for (int i = 0; i < desc_thread_count; i++) {
        double throughput = (double)submit_args[i].submit_cnt / (double)submit_args[i].elapsed_ns * (double)1000.0;
        printf("  thread %2d: submitted=%d  time=%llu ns  throughput=%.2f MTasks/s\n",
               submit_args[i].thread_id,
               submit_args[i].submit_cnt,
               (unsigned long long)submit_args[i].elapsed_ns,
               throughput);
        total_submit_cnt += submit_args[i].submit_cnt;
        if (submit_args[i].elapsed_ns > submit_max_ns)
            submit_max_ns = submit_args[i].elapsed_ns;
    }

    uint64_t submit_elapsed = end_ns - desc_end_ns;
    printf("\nphase timing:\n");
    printf("  alloc+desc phase: %llu ns\n", (unsigned long long)(desc_end_ns - start_ns));
    printf("  submit phase:     %llu ns\n", (unsigned long long)submit_elapsed);
    printf("  desc max thread time:   %llu ns\n", (unsigned long long)desc_max_ns);
    printf("  submit max thread time: %llu ns\n", (unsigned long long)submit_max_ns);
    printf("orchestrator total elapsed time (1 alloc + %d desc + %d submit threads): %llu ns\n",
           desc_thread_count, desc_thread_count, (unsigned long long)elapsed_ns);
    printf("desc=%d  submit=%d\n", total_cnt, total_submit_cnt);
    printf("throughput = %d / (%llu + %llu) ns = %.2f MTasks/s\n",
           total_tasks,
           (unsigned long long)desc_max_ns,
           (unsigned long long)submit_max_ns,
           (desc_max_ns + submit_max_ns) > 0
               ? (double)total_tasks * 1000.0 / (double)(desc_max_ns + submit_max_ns)
               : 0.0);

    const char *dump_path = getenv("DEP_DUMP_PATH");
    if (dump_path && dump_path[0]) {
        dump_predecessors(dump_path, total_tasks);
        printf("wrote predecessor dump: %s\n", dump_path);
    }

    free(submit_args);
    free(submit_threads);
    free(desc_args);
    free(desc_threads);
    return 0;
}
