/* Thin wrapper: includes only the submit case header and exposes callable
 * functions for use by orchestrator.c.
 *
 * Initializes per-thread tensormaps and predecessor ring buffer. */

#include "tensormap.h"
#include "log.h"
#include "qwen3_14b_decoder_submit.h"

extern void init_predecessors(void);

void orc_submit_init(int thread_count)
{
    ring_buf_init();
    init_predecessors();
    for (int i = 0; i < thread_count && i < SUBMIT_MAX_THREADS; i++) {
        tm_pt_init(&g_tm_pt[i]);
    }
}

void orc_submit_cleanup(void)
{
}

int orc_submit_call(int thread_id, int total_tasks, int *submit_cnt)
{
    orchestrator_submit(thread_id, total_tasks, submit_cnt);
    return *submit_cnt;
}
