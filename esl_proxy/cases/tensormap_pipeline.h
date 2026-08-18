#ifndef TENSORMAP_PIPELINE_H
#define TENSORMAP_PIPELINE_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ring_buf.h"
#include "tensormap_core.h"

#ifndef NUM_ORCH_TILES
#define NUM_ORCH_TILES 6
#endif

#ifndef TMD_POOL_SIZE
#define TMD_POOL_SIZE 1024u
#endif
#ifndef TMD_NUM_BUCKETS
#define TMD_NUM_BUCKETS 256u
#endif
#ifndef TMD_TASK_WINDOW
#define TMD_TASK_WINDOW 1024u
#endif

#define TMD_BUF_BYTES_PIPE                                                 \
    (sizeof(TmHeader) + (uint64_t)TMD_NUM_BUCKETS * sizeof(int32_t) +     \
        TM_ENTRY_ALIGN + (uint64_t)TMD_POOL_SIZE * sizeof(TmEntry) +      \
        (uint64_t)TMD_POOL_SIZE * sizeof(int32_t) +                       \
        (uint64_t)TMD_TASK_WINDOW * sizeof(int32_t))

enum {
    TM_PEND_IN = 1u,
    TM_PEND_OUT = 2u,
    TM_PEND_INOUT = (TM_PEND_IN | TM_PEND_OUT)
};

#ifndef TM_PENDING_MAX_IO
#define TM_PENDING_MAX_IO 16u
#endif
#ifndef TM_PENDING_MAX_PRED
#define TM_PENDING_MAX_PRED 64u
#endif

#ifndef TOTAL_TASKS
#define TOTAL_TASKS 3096
#endif

#define PIPE_ARENA_SIZE 4096

typedef struct {
    const Tensor *t;
    uint8_t kind;
} TmPendingSlot;

typedef struct {
    TmTensorMap map;
    _Alignas(TM_ENTRY_ALIGN) uint8_t buf[TMD_BUF_BYTES_PIPE];
    TmPendingSlot pend[TM_PENDING_MAX_IO];
    int pend_n;
} TmDepsStatePipe;

static TmDepsStatePipe g_tm_deps_pipe[NUM_ORCH_TILES];

typedef struct {
    int cnt;
    char _pad[64 - sizeof(int)];
} AlignedSubtaskCnt;

static AlignedSubtaskCnt g_subtask_cnt_per_tile[NUM_ORCH_TILES];

typedef struct {
    uint32_t cnt;
    uint32_t *exp;
} PredEntry;

static PredEntry g_pred_local[NUM_ORCH_TILES][TOTAL_TASKS];

#define PIPE_PRED_BUFF_SIZE 16384
static uint32_t *g_pred_ring_buf[NUM_ORCH_TILES];

typedef struct {
    uint32_t *tail;
    char _pad[64 - sizeof(uint32_t*)];
} AlignedPredTail;

static AlignedPredTail g_pred_ring_tail[NUM_ORCH_TILES];

static inline void pred_ring_init_tile(int tile) {
    g_pred_ring_buf[tile] = (uint32_t *)malloc(
        (size_t)PIPE_PRED_BUFF_SIZE * sizeof(uint32_t));
    g_pred_ring_tail[tile].tail = g_pred_ring_buf[tile];
}

static inline void pred_ring_deinit_tile(int tile) {
    free(g_pred_ring_buf[tile]);
    g_pred_ring_buf[tile] = NULL;
    g_pred_ring_tail[tile].tail = NULL;
}

typedef struct {
    uint32_t consumer;
    uint32_t preds[TM_PENDING_MAX_PRED];
    int pn;
    bool is_inout;
    int tile;
} TmCollectCtxPipe;

static inline bool tm_collect_on_match_pipe(TmEntry *e, TmOverlap ov, void *ctx) {
    TmCollectCtxPipe *c = (TmCollectCtxPipe *)ctx;
    const uint32_t p = (uint32_t)tm_local_of(e->producer_id);
    if (p != c->consumer) {
        for (int i = 0; i < c->pn; i++) {
            if (c->preds[i] == p) goto after_pred;
        }
        if (c->pn < TM_PENDING_MAX_PRED) {
            c->preds[c->pn++] = p;
        }
    }
after_pred:
    if (c->is_inout && ov == TM_OVERLAP_COVERED) {
        tm_remove(&g_tm_deps_pipe[c->tile].map, e);
    }
    return true;
}

static inline void tm_deps_init_pipe(int tile) {
    TmConfig cfg;
    cfg.num_buckets = TMD_NUM_BUCKETS;
    cfg.pool_size = TMD_POOL_SIZE;
    cfg.num_rings = 1;
    cfg.task_window[0] = TMD_TASK_WINDOW;
    for (uint32_t r = 1; r < TM_MAX_RINGS; r++) {
        cfg.task_window[r] = 1;
    }
    tm_init(&g_tm_deps_pipe[tile].map, g_tm_deps_pipe[tile].buf, &cfg);
    g_tm_deps_pipe[tile].pend_n = 0;
}

static inline void tm_pending_push_pipe(int tile, const Tensor *t, uint8_t kind) {
    TmDepsStatePipe *ds = &g_tm_deps_pipe[tile];
    if (ds->pend_n < (int)TM_PENDING_MAX_IO) {
        ds->pend[ds->pend_n].t = t;
        ds->pend[ds->pend_n].kind = kind;
        ds->pend_n++;
    }
}

static inline void new_task_pipe(uint32_t task_id, uint32_t type,
                                  uint32_t count, uint32_t duration, int tile) {
    (void)type;
    struct task_desc *td = &g_basic_buf[task_id & RING_MASK];
    if (count > 1)
        td->mode = ORG_MODE_SPMD_SYNC;
    td->count = count;
    td->duration = duration;
    td->tensor_cnt = 0;
    td->scalar_cnt = 0;
    g_subtask_cnt_per_tile[tile].cnt += count;
}

#define new_task(task_id, type, count, duration) \
    new_task_pipe((task_id), (type), (count), (duration), __pipe_tile)

static inline void tm_in_pipe_ptr(int tile, uint32_t tid, const Tensor *t) {
    struct task_desc *td = &g_basic_buf[tid & RING_MASK];
    td->data[td->tensor_cnt++] = t->buffer_addr;
    tm_pending_push_pipe(tile, t, TM_PEND_IN);
}

static inline void tm_out_pipe_ptr(int tile, uint32_t tid, const Tensor *t) {
    struct task_desc *td = &g_basic_buf[tid & RING_MASK];
    td->data[td->tensor_cnt++] = t->buffer_addr;
    tm_pending_push_pipe(tile, t, TM_PEND_OUT);
}

static inline void tm_inout_pipe_ptr(int tile, uint32_t tid, const Tensor *t) {
    struct task_desc *td = &g_basic_buf[tid & RING_MASK];
    td->data[td->tensor_cnt++] = t->buffer_addr;
    tm_pending_push_pipe(tile, t, TM_PEND_INOUT);
}

static inline void tm_ro_pipe_ptr(uint32_t tid, const Tensor *t) {
    struct task_desc *td = &g_basic_buf[tid & RING_MASK];
    td->data[td->tensor_cnt++] = t->buffer_addr;
}

#define tm_in(tid, t)       tm_in_pipe_ptr(__pipe_tile, (tid), &(t))
#define tm_out(tid, t)      tm_out_pipe_ptr(__pipe_tile, (tid), &(t))
#define tm_inout(tid, t)    tm_inout_pipe_ptr(__pipe_tile, (tid), &(t))
#define tm_in_ro(tid, t)    tm_ro_pipe_ptr((tid), &(t))
#define tm_out_ro(tid, t)   tm_ro_pipe_ptr((tid), &(t))
#define tm_inout_ro(tid, t) tm_ro_pipe_ptr((tid), &(t))

static inline void tm_fast_lookup(TmTensorMap *self, const Tensor *t,
    TmMatchFn on_match, void *ctx) {
    const uint32_t b = tm_hash(self, t->buffer_addr);
    int32_t cur = tm_buckets(self)[b];
    TmEntry *pl = tm_pool(self);

    while (cur != -1) {
        TmEntry *e = &pl[cur];
        if (e->base_addr == t->buffer_addr) {
            const TmOverlap st = tm_check_overlap(t, e);
            if (st != TM_OVERLAP_NONE && !on_match(e, st, ctx))
                return;
        }
        cur = e->next_in_bucket;
    }
}

static inline void tm_submit_pipe(int tile, uint32_t tid) {
    TmDepsStatePipe *ds = &g_tm_deps_pipe[tile];
    TmTensorMap *map = &ds->map;

    TmCollectCtxPipe ctx = {.consumer = tid, .pn = 0, .tile = tile};

    for (int i = 0; i < ds->pend_n; i++) {
        if (ds->pend[i].kind & TM_PEND_IN) {
            ctx.is_inout = (ds->pend[i].kind == TM_PEND_INOUT);
            tm_fast_lookup(map, ds->pend[i].t, tm_collect_on_match_pipe, &ctx);
        }
    }

    struct predecessor_list *ptr = &g_predecessors[tid];
    if (ctx.pn > 0) {
        uint32_t *start = g_pred_ring_tail[tile].tail;
        g_pred_ring_tail[tile].tail += (uint32_t)ctx.pn;
        ptr->exp = start;
        for (int i = 0; i < ctx.pn; i++)
            start[i] = ctx.preds[i];
        ptr->cnt = ctx.pn;
    } else {
        ptr->cnt = 0;
        ptr->exp = NULL;
    }

    for (int i = 0; i < ds->pend_n; i++) {
        if (ds->pend[i].kind & TM_PEND_OUT) {
            tm_insert_tensor(map, ds->pend[i].t, tid);
        }
    }

    ds->pend_n = 0;
}

static inline void add_scalar_pipe(uint32_t task_id, int64_t v) {
    struct task_desc *td = &g_basic_buf[task_id & RING_MASK];
    td->scalar[td->scalar_cnt++] = v;
}
#define add_scalar(task_id, v) add_scalar_pipe((task_id), (v))

#define tm_submit(tid) tm_submit_pipe(__pipe_tile, (tid))

#endif /* TENSORMAP_PIPELINE_H */
