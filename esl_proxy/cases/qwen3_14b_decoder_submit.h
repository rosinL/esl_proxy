// Per-thread filtered tensormap — fully independent, zero-lock, zero-barrier.
//
// Each thread:
//   1. Collect: copy own task descs into a contiguous local array
//      (eliminates strided access to 12.6MB shared g_task_tensor_buf)
//   2. Pre-scan: collect base_addr set from own inputs
//   3. Filtered insert: insert ONLY outputs whose base_addr is in the set
//      → map has ~500-1000 entries → fits in L2
//   4. Lookup: query own inputs in private L2-resident map
//
// Two tensors can only overlap if they share the same base_addr (views of
// the same allocation). Filtering by base_addr never misses a dependency.
//
// No shared mutable state → no locks, no barrier, no cross-map lookup.
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

#include "tensormap.h"
#include "orch_config.h"
#include "spin.h"

extern struct task_tensor_desc g_task_tensor_buf[RING_SIZE];
extern int desc_thread_count;
extern int desc_batch_size;

/* -----------------------------------------------------------------------
 * Per-thread tensormap storage
 *
 * pool_size: filtered map has ~500-1000 entries. 2048 gives headroom.
 *   2048 × 128B = 256KB → fits in L2.
 * --------------------------------------------------------------------- */
#ifndef TM_PT_POOL_SIZE
#define TM_PT_POOL_SIZE 2048u
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

/* -----------------------------------------------------------------------
 * Base_addr hash set — open addressing, linear probing.
 *
 * Two tensors can only overlap if they share base_addr. This set filters
 * which outputs to insert: only outputs whose base_addr matches one of
 * this thread's input base_addrs can possibly overlap.
 * --------------------------------------------------------------------- */
#define ADDR_SET_BITS 9
#define ADDR_SET_SIZE (1u << ADDR_SET_BITS)
#define ADDR_SET_MASK (ADDR_SET_SIZE - 1u)

typedef struct {
    uint64_t slots[ADDR_SET_SIZE];
} AddrSet;

static inline void addrset_init(AddrSet *s)
{
    for (uint32_t i = 0; i < ADDR_SET_SIZE; i++) {
        s->slots[i] = 0;
    }
}

static inline void addrset_add(AddrSet *s, uint64_t addr)
{
    if (addr == 0) return;
    uint32_t i = (uint32_t)((addr * 0x9E3779B97F4A7C15ULL) >> (64 - ADDR_SET_BITS));
    for (uint32_t k = 0; k < ADDR_SET_SIZE; k++) {
        uint32_t idx = (i + k) & ADDR_SET_MASK;
        if (s->slots[idx] == addr) return;
        if (s->slots[idx] == 0) {
            s->slots[idx] = addr;
            return;
        }
    }
}

static inline bool addrset_contains(const AddrSet *s, uint64_t addr)
{
    if (addr == 0) return false;
    uint32_t i = (uint32_t)((addr * 0x9E3779B97F4A7C15ULL) >> (64 - ADDR_SET_BITS));
    for (uint32_t k = 0; k < ADDR_SET_SIZE; k++) {
        uint32_t idx = (i + k) & ADDR_SET_MASK;
        if (s->slots[idx] == addr) return true;
        if (s->slots[idx] == 0) return false;
    }
    return false;
}

/* -----------------------------------------------------------------------
 * Non-atomic single-writer tensormap operations.
 * --------------------------------------------------------------------- */
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

/* -----------------------------------------------------------------------
 * Collector callback — deduplicates predecessors, NO INOUT removal.
 * --------------------------------------------------------------------- */
static inline bool tm_collect_safe(TmEntry *e, TmOverlap ov, void *ctx)
{
    (void)ov;
    TmCollectCtx *c = (TmCollectCtx *)ctx;
    const uint32_t p = (uint32_t)tm_local_of(e->producer_id);
    if (p != c->consumer) {
        for (int i = 0; i < c->pn; i++) {
            if (c->preds[i] == p) {
                return true;
            }
        }
        if (c->pn < (int)TM_PENDING_MAX_PRED) {
            c->preds[c->pn++] = p;
        }
    }
    return true;
}

/* -----------------------------------------------------------------------
 * Task ownership — same strided partitioning as DESC_DO_OR_SKIP.
 * --------------------------------------------------------------------- */
static inline int submit_owns(int tid, int thread_id)
{
    int batch = tid / desc_batch_size;
    return (batch % desc_thread_count) == thread_id;
}

/* -----------------------------------------------------------------------
 * Compact owned-task descriptor for local collection.
 * Stores task_id + pointers to the shared desc's IO arrays.
 * --------------------------------------------------------------------- */
typedef struct {
    uint32_t tid;
    const struct task_tensor_desc *desc;
} OwnedTask;

/* -----------------------------------------------------------------------
 * Parallel tm_submit entry point — fully independent per thread.
 * --------------------------------------------------------------------- */
void orchestrator_submit(int thread_id, int total_tasks, int *submit_cnt)
{
    TmTensorMap *map = &g_tm_pt[thread_id].map;
    AddrSet addr_set;
    addrset_init(&addr_set);

    /* === Step 1: Collect owned tasks into contiguous local array ===
     * One pass over g_task_tensor_buf (strided read), then all subsequent
     * accesses to task metadata are sequential over the local array. */
    OwnedTask *owned = (OwnedTask *)malloc((size_t)total_tasks * sizeof(OwnedTask));
    int n_owned = 0;
    for (int tid = 0; tid < total_tasks; tid++) {
        if (submit_owns(tid, thread_id)) {
            owned[n_owned].tid = (uint32_t)tid;
            owned[n_owned].desc = &g_task_tensor_buf[(uint32_t)tid & RING_MASK];
            n_owned++;
        }
    }

    /* === Step 2: Pre-scan — collect base_addr set from own inputs === */
    for (int i = 0; i < n_owned; i++) {
        const struct task_tensor_desc *desc = owned[i].desc;
        for (int j = 0; j < desc->in_cnt; j++) {
            addrset_add(&addr_set, desc->in_data[j].buffer_addr);
        }
        for (int j = 0; j < desc->inout_cnt; j++) {
            addrset_add(&addr_set, desc->inout_data[j].buffer_addr);
        }
    }

    /* === Step 3: Filtered insert — only outputs matching input base_addrs ===
     * Sequential scan over all tasks' outputs. addrset_contains is O(1).
     * Skips ~70% of outputs that can never match → smaller map → L2 resident. */
    for (int tid = 0; tid < total_tasks; tid++) {
        const struct task_tensor_desc *desc = &g_task_tensor_buf[(uint32_t)tid & RING_MASK];
        for (int j = 0; j < desc->out_cnt; j++) {
            if (addrset_contains(&addr_set, desc->out_data[j].buffer_addr)) {
                tm_pt_insert(map, &desc->out_data[j], (uint32_t)tid);
            }
        }
        for (int j = 0; j < desc->inout_cnt; j++) {
            if (addrset_contains(&addr_set, desc->inout_data[j].buffer_addr)) {
                tm_pt_insert(map, &desc->inout_data[j], (uint32_t)tid);
            }
        }
    }

    /* === Step 4: Lookup own tasks' inputs in private (small) map ===
     * Sequential access over local owned array. Map is L2-resident. */
    for (int i = 0; i < n_owned; i++) {
        uint32_t tid = owned[i].tid;
        const struct task_tensor_desc *desc = owned[i].desc;
        TmCollectCtx ctx = {.consumer = tid, .pn = 0};

        for (int j = 0; j < desc->in_cnt; j++) {
            tm_pt_lookup(map, &desc->in_data[j], tm_collect_safe, &ctx);
        }
        for (int j = 0; j < desc->inout_cnt; j++) {
            tm_pt_lookup(map, &desc->inout_data[j], tm_collect_safe, &ctx);
        }

        if (ctx.pn > 0) {
            add_predecessors(tid, ctx.preds, (uint32_t)ctx.pn, 0);
        }
    }

    free(owned);
    *submit_cnt = n_owned;
}
