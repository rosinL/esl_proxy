// Per-thread tensormap with streaming lookup→insert — zero-lock, zero-barrier.
//
// Each thread processes ALL tasks in task_id order, preserving the original
// tm_submit's lookup→insert ordering per task. This is required for correct
// INOUT removal: a consumer's lookup sees only producers with smaller
// task_ids, and tm_remove deletes covered entries so later consumers see
// only the latest writer.
//
// For non-owned tasks: only INOUT lookup (for removal) + insert outputs.
// For owned tasks: full lookup + add_predecessors + insert outputs.
//
// Each thread's map is private (single-writer) → no locks, no atomics.
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <sched.h>

#include "tensormap.h"
#include "orch_config.h"
#include "spin.h"

extern struct task_tensor_desc g_task_tensor_buf[RING_SIZE];
extern int desc_thread_count;
extern int desc_batch_size;

#ifndef TM_PT_POOL_SIZE
#define TM_PT_POOL_SIZE 8192u
#endif
#ifndef TM_PT_TASK_WINDOW
#define TM_PT_TASK_WINDOW 4096u
#endif

#define TM_PT_BUF_BYTES                                                       \
    (sizeof(TmHeader) + (uint64_t)TMD_NUM_BUCKETS * sizeof(int32_t) +         \
     TM_ENTRY_ALIGN + (uint64_t)TM_PT_POOL_SIZE * sizeof(TmEntry) +           \
     (uint64_t)TM_PT_POOL_SIZE * sizeof(int32_t) +                            \
     (uint64_t)TM_PT_TASK_WINDOW * sizeof(int32_t))

typedef struct {
    TmTensorMap map;
    _Alignas(128) uint8_t buf[TM_PT_BUF_BYTES];
} TmPtState;

#ifndef SUBMIT_MAX_THREADS
#define SUBMIT_MAX_THREADS 128
#endif

static TmPtState g_tm_pt[SUBMIT_MAX_THREADS];

static inline void tm_pt_init(TmPtState *state)
{
    TmConfig cfg;
    cfg.num_buckets = TMD_NUM_BUCKETS;
    cfg.pool_size = TM_PT_POOL_SIZE;
    cfg.num_rings = 1;
    cfg.task_window[0] = TM_PT_TASK_WINDOW;
    for (uint32_t r = 1; r < TM_MAX_RINGS; r++) {
        cfg.task_window[r] = 1;
    }
    tm_init(&state->map, state->buf, &cfg);
}

static inline int32_t tm_pt_new_entry(TmTensorMap *self)
{
    TmHeader *h = tm_hdr(self);
    if (h->free_num > 0) {
        return tm_free_list(self)[--h->free_num];
    }
    if (h->next_entry_idx >= (int32_t)h->cfg.pool_size) {
        return -1;
    }
    return h->next_entry_idx++;
}

static inline void tm_pt_insert(TmTensorMap *self, const Tensor *t,
                                uint32_t tid)
{
    const int32_t idx = tm_pt_new_entry(self);
    if (idx < 0) {
        return;
    }
    TmEntry *e = &tm_pool(self)[idx];
    tm_copy_tensor_to_entry(t, e);
    tm_link_entry(self, idx, t->buffer_addr, tm_make_id(0, tid));
}

static inline void tm_pt_lookup(TmTensorMap *self, const Tensor *t,
                                TmMatchFn on_match, void *ctx)
{
    const uint32_t b = tm_hash(self, t->buffer_addr);
    int32_t cur = tm_buckets(self)[b];
    TmEntry *pl = tm_pool(self);

    while (cur != -1) {
        const int32_t next = pl[cur].next_in_bucket;
        if (next != -1) {
            __builtin_prefetch(&pl[next], 0, 0);
        }
        TmEntry *e = &pl[cur];
        if (tm_entry_valid(self, e) && e->base_addr == t->buffer_addr) {
            const TmOverlap st = tm_check_overlap(t, e);
            if (st != TM_OVERLAP_NONE && !on_match(e, st, ctx)) {
                return;
            }
        }
        cur = next;
    }
}

typedef struct {
    uint32_t consumer;
    uint32_t preds[TM_PENDING_MAX_PRED];
    int pn;
    bool is_inout;
    TmTensorMap *map;
} SubmitCollectCtx;

static inline bool tm_collect_safe(TmEntry *e, TmOverlap ov, void *ctx)
{
    SubmitCollectCtx *c = (SubmitCollectCtx *)ctx;
    const uint32_t p = (uint32_t)tm_local_of(e->producer_id);
    if (p != c->consumer) {
        for (int i = 0; i < c->pn; i++) {
            if (c->preds[i] == p) {
                goto after_pred;
            }
        }
        if (c->pn < (int)TM_PENDING_MAX_PRED) {
            c->preds[c->pn++] = p;
        }
    }
after_pred:
    if (c->is_inout && ov == TM_OVERLAP_COVERED) {
        tm_remove(c->map, e);
    }
    return true;
}

static inline bool tm_collect_remove_only(TmEntry *e, TmOverlap ov, void *ctx)
{
    SubmitCollectCtx *c = (SubmitCollectCtx *)ctx;
    if (c->is_inout && ov == TM_OVERLAP_COVERED) {
        tm_remove(c->map, e);
    }
    return true;
}

static inline int submit_owns(int tid, int thread_id)
{
    int batch = tid / desc_batch_size;
    return (batch % desc_thread_count) == thread_id;
}

void orchestrator_submit(int thread_id, int total_tasks, int *submit_cnt)
{
    TmTensorMap *map = &g_tm_pt[thread_id].map;
    int n_owned = 0;

    for (int tid = 0; tid < total_tasks; tid++) {
        while (!atomic_load_explicit(&g_task_ready[(uint32_t)tid & RING_MASK],
                                     memory_order_acquire)) {
            sched_yield();
        }
        const struct task_tensor_desc *desc = &g_task_tensor_buf[(uint32_t)tid & RING_MASK];
        bool is_owned = submit_owns(tid, thread_id);

        if (is_owned) {
            n_owned++;
            SubmitCollectCtx ctx = {.consumer = (uint32_t)tid, .pn = 0, .map = map};

            for (int j = 0; j < desc->in_cnt; j++) {
                ctx.is_inout = false;
                tm_pt_lookup(map, &desc->in_data[j], tm_collect_safe, &ctx);
            }
            for (int j = 0; j < desc->inout_cnt; j++) {
                ctx.is_inout = true;
                tm_pt_lookup(map, &desc->inout_data[j], tm_collect_safe, &ctx);
            }

            if (ctx.pn > 0) {
                add_predecessors((uint32_t)tid, ctx.preds, (uint32_t)ctx.pn, 0);
            }
        } else {
            for (int j = 0; j < desc->inout_cnt; j++) {
                SubmitCollectCtx ctx = {.consumer = (uint32_t)tid, .pn = 0,
                                        .is_inout = true, .map = map};
                tm_pt_lookup(map, &desc->inout_data[j], tm_collect_remove_only, &ctx);
            }
        }

        for (int j = 0; j < desc->out_cnt; j++) {
            tm_pt_insert(map, &desc->out_data[j], (uint32_t)tid);
        }
        for (int j = 0; j < desc->inout_cnt; j++) {
            tm_pt_insert(map, &desc->inout_data[j], (uint32_t)tid);
        }
    }

    *submit_cnt = n_owned;
}
