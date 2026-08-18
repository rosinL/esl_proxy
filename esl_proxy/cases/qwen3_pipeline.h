#ifndef QWEN3_PIPELINE_H
#define QWEN3_PIPELINE_H

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "mem_pool.h"
#include "ring_buf.h"
#include "spin.h"
#include "tensor.h"
#include "pipeline_spin.h"
#include "tensormap_pipeline.h"
#include "qwen3_pipeline_stats.h"

#define DUR_RMSNORM 23950
#define DUR_Q_PROJ 26060
#define DUR_K_PROJ 18170
#define DUR_V_PROJ 17890
#define DUR_QK_NORM 13190
#define DUR_ROPE_KV_CACHE 9480
#define DUR_QK_MATMUL 29350
#define DUR_SOFTMAX 19400
#define DUR_SV_MATMUL 31650
#define DUR_ONLINE_SOFTMAX 20820
#define DUR_OUT_PROJ 40750
#define DUR_POST_RMSNORM 24390
#define DUR_GATE_PROJ 95700
#define DUR_UP_PROJ 97140
#define DUR_SILU 2820
#define DUR_DOWN_PROJ 72220
#define DUR_DOWN_PROJ_RES 2590

int g_subtask_cnt = 0;

static inline int qwen3_min_i(int a, int b) { return a < b ? a : b; }

static inline int qwen3_blocks_per_task(int total_chunks) {
    static const int targets[5] = {1, 2, 4, 8, 1 << 30};
    int target = targets[QWEN3_SPMD_TIER];
    return qwen3_min_i(total_chunks, target);
}

static inline int qwen3_cur_blocks(int total_chunks, int base) {
    return qwen3_min_i(qwen3_blocks_per_task(total_chunks), total_chunks - base);
}

#define P1_PER_TILE  38
#define P1_TOTAL     228
#define P2_PER_TOKEN 17
#define P2_TOTAL     1530
#define P3_PER_TILE  223
#define P3_TOTAL     1338
#define TOTAL_TASKS  3096

static inline uint32_t p1_tid(int tile, int i) {
    return (uint32_t)(tile * P1_PER_TILE + i);
}
static inline uint32_t p2_tid(int tile, int b_local, int s) {
    int b = tile * 16 + b_local;
    return (uint32_t)(P1_TOTAL + b * P2_PER_TOKEN + s);
}
static inline uint32_t p3_tid(int tile, int i) {
    return (uint32_t)(P1_TOTAL + P2_TOTAL + tile * P3_PER_TILE + i);
}

static inline void pin_thread(int core_id) {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
#endif
}

typedef struct {
    atomic_int val;
    char _pad[64 - sizeof(atomic_int)];
} cacheline_atomic_int;

static atomic_int g_start_barrier = 0;
static atomic_int g_threads_at_barrier = 0;
static atomic_int g_submit_done_cnt = 0;

static Tensor ext_hidden_states;
static Tensor ext_input_rms_weight;
static Tensor ext_wq;
static Tensor ext_wk;
static Tensor ext_wv;
static Tensor ext_q_norm_weight;
static Tensor ext_k_norm_weight;
static Tensor ext_seq_lens;
static Tensor ext_block_table;
static Tensor ext_slot_mapping;
static Tensor ext_rope_cos;
static Tensor ext_rope_sin;
static Tensor ext_k_cache;
static Tensor ext_v_cache;
static Tensor ext_wo;
static Tensor ext_post_rms_weight;
static Tensor ext_w_gate;
static Tensor ext_w_up;
static Tensor ext_w_down;
static Tensor ext_out;

static Tensor g_q_proj;
static Tensor g_k_proj;
static Tensor g_v_proj;
static Tensor g_q_proj_norm;
static Tensor g_k_proj_norm;
static Tensor g_attn_out[NUM_ORCH_TILES];

typedef struct {
    Tensor raw_scores;
    Tensor exp_padded;
    Tensor cur_mi;
    Tensor cur_li;
    Tensor oi_tmp;
    Tensor q_padded_local;
    Tensor k_cache_update;
    Tensor v_cache_update;
} TokenTensors;

typedef struct {
    Tensor normed_tile;
    TokenTensors tokens[16];
    int num_tokens;
    Tensor resid1_tile;
    Tensor gm_pipe_buffer_0;
    Tensor post_norm_tile;
    Tensor mlp_tile;
    Tensor gate_tile;
    Tensor up_tile;
    Tensor down_tile;
} TileTensors;

static TileTensors g_tiles[NUM_ORCH_TILES];

static void alloc_phase1_tensors(int tile) {
    g_tiles[tile].normed_tile =
        alloc_tensors((uint32_t[2]){16, 5120}, 2, BFLOAT16);
}

static void alloc_phase2_tokens(int tile) {
    int n = (tile < 5) ? 16 : 10;
    g_tiles[tile].num_tokens = n;
    for (int b_local = 0; b_local < n; b_local++) {
        TokenTensors *tk = &g_tiles[tile].tokens[b_local];
        tk->raw_scores    = alloc_tensors((uint32_t[2]){4096, 128}, 2, FLOAT32);
        tk->exp_padded    = alloc_tensors((uint32_t[2]){4096, 128}, 2, BFLOAT16);
        tk->cur_mi        = alloc_tensors((uint32_t[2]){4096, 1}, 2, FLOAT32);
        tk->cur_li        = alloc_tensors((uint32_t[2]){4096, 1}, 2, FLOAT32);
        tk->oi_tmp        = alloc_tensors((uint32_t[2]){4096, 128}, 2, FLOAT32);
        tk->q_padded_local= alloc_tensors((uint32_t[2]){128, 128}, 2, BFLOAT16);
        tk->k_cache_update= alloc_tensors((uint32_t[2]){8, 128}, 2, BFLOAT16);
        tk->v_cache_update= alloc_tensors((uint32_t[2]){8, 128}, 2, BFLOAT16);
    }
}

static void alloc_phase3_tensors(int tile) {
    TileTensors *t = &g_tiles[tile];
    t->resid1_tile     = alloc_tensors((uint32_t[2]){16, 5120}, 2, FLOAT32);
    t->gm_pipe_buffer_0= alloc_tensors((uint32_t[2]){16384, 40}, 2, FLOAT32);
    t->post_norm_tile  = alloc_tensors((uint32_t[2]){16, 5120}, 2, BFLOAT16);
    t->mlp_tile        = alloc_tensors((uint32_t[2]){16, 17408}, 2, BFLOAT16);
    t->gate_tile       = alloc_tensors((uint32_t[2]){16, 17408}, 2, FLOAT32);
    t->up_tile         = alloc_tensors((uint32_t[2]){16, 17408}, 2, FLOAT32);
    t->down_tile       = alloc_tensors((uint32_t[2]){16, 5120}, 2, FLOAT32);
}

static int desc_phase1(int tile) {
    int __pipe_tile = tile;
    TileTensors *t = &g_tiles[tile];
    int64_t b0 = tile * 16;
    int64_t cur_valid = (tile < 5) ? 16 : (90 - 80);
    uint32_t tid = p1_tid(tile, 0);
    int count = 0;

    new_task(tid, TASK_TYPE_VECTOR, 1, DUR_RMSNORM);
    tm_in_ro(tid, ext_hidden_states);
    tm_out(tid, t->normed_tile);
    tm_in_ro(tid, ext_input_rms_weight);
    add_scalar(tid, b0);
    add_scalar(tid, cur_valid);
    tm_submit(tid);
    count++; tid++;

    for (int base = 0; base < 20; base += qwen3_blocks_per_task(20)) {
        int cur_blocks = qwen3_cur_blocks(20, base);
        new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_Q_PROJ);
        Tensor q_piece = view(g_q_proj, (uint32_t)b0, base * 256u, 16u, cur_blocks * 256u);
        tm_in(tid, t->normed_tile);
        tm_in_ro(tid, ext_wq);
        tm_out(tid, q_piece);
        add_scalar(tid, b0);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;
    }

    for (int base = 0; base < 8; base += qwen3_blocks_per_task(8)) {
        int cur_blocks = qwen3_cur_blocks(8, base);
        new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_K_PROJ);
        tm_in(tid, t->normed_tile);
        tm_in_ro(tid, ext_wk);
        Tensor k_piece = view(g_k_proj, (uint32_t)b0, base * 128u, 16u, cur_blocks * 128u);
        tm_out(tid, k_piece);
        add_scalar(tid, b0);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;

        new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_V_PROJ);
        tm_in(tid, t->normed_tile);
        tm_in_ro(tid, ext_wv);
        Tensor v_piece = view(g_v_proj, (uint32_t)b0, base * 128u, 16u, cur_blocks * 128u);
        tm_out(tid, v_piece);
        add_scalar(tid, b0);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;
    }

    new_task(tid, TASK_TYPE_VECTOR, 1, DUR_QK_NORM);
    Tensor k0_norm = view(g_k_proj_norm, (uint32_t)b0, 0u, 16u, 1024u);
    Tensor q0_norm = view(g_q_proj_norm, (uint32_t)b0, 0u, 16u, 5120u);
    Tensor q0_in = view(g_q_proj, (uint32_t)b0, 0u, 16u, 5120u);
    Tensor k0_in = view(g_k_proj, (uint32_t)b0, 0u, 16u, 1024u);
    tm_out(tid, k0_norm);
    tm_out(tid, q0_norm);
    tm_in(tid, q0_in);
    tm_in_ro(tid, ext_q_norm_weight);
    tm_in_ro(tid, ext_k_norm_weight);
    tm_in(tid, k0_in);
    tm_submit(tid);
    count++; tid++;

    return count;
}

static int desc_phase2(int tile) {
    int __pipe_tile = tile;
    TileTensors *t = &g_tiles[tile];
    int n = t->num_tokens;
    int count = 0;

    for (int b_local = 0; b_local < n; b_local++) {
        int64_t b = tile * 16 + b_local;
        TokenTensors *tk = &t->tokens[b_local];
        int64_t b_tile0 = tile * 16;
        int64_t slot = b;
        int64_t slot_block = slot / 128;
        int64_t slot_offset = slot - slot_block * 128;
        uint32_t tid = p2_tid(tile, b_local, 0);

        Tensor k_cache_local = view(ext_k_cache, (uint32_t)(b * 8u), 0u, 8u, 128u);
        Tensor v_cache_local = view(ext_v_cache, (uint32_t)(b * 8u), 0u, 8u, 128u);

        new_task(tid, TASK_TYPE_VECTOR, 1, DUR_ROPE_KV_CACHE);
        Tensor k0_norm = view(g_k_proj_norm, (uint32_t)b_tile0, 0u, 16u, 1024u);
        Tensor v0 = view(g_v_proj, (uint32_t)b_tile0, 0u, 16u, 1024u);
        Tensor q0_norm = view(g_q_proj_norm, (uint32_t)b_tile0, 0u, 16u, 5120u);
        tm_out(tid, tk->q_padded_local);
        tm_in_ro(tid, k_cache_local);
        tm_in_ro(tid, v_cache_local);
        tm_out(tid, tk->k_cache_update);
        tm_out(tid, tk->v_cache_update);
        tm_in(tid, k0_norm);
        tm_in_ro(tid, ext_rope_cos);
        tm_in_ro(tid, ext_rope_sin);
        tm_in_ro(tid, ext_rope_cos);
        tm_in_ro(tid, ext_rope_sin);
        tm_in(tid, v0);
        tm_in(tid, q0_norm);
        add_scalar(tid, slot_block);
        add_scalar(tid, slot_offset);
        add_scalar(tid, b);
        tm_submit(tid);
        count++; tid++;

        for (int base = 0; base < 4; base += qwen3_blocks_per_task(4)) {
            int cur_blocks = qwen3_cur_blocks(4, base);
            Tensor row_piece = view(tk->raw_scores, base * 1024u, 0u,
                                    (uint32_t)(cur_blocks * 1024), 128u);

            new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_QK_MATMUL);
            tm_in(tid, tk->q_padded_local);
            tm_out(tid, row_piece);
            tm_in_ro(tid, ext_block_table);
            tm_in(tid, tk->k_cache_update);
            add_scalar(tid, b);
            add_scalar(tid, 8);
            add_scalar(tid, b * 32);
            add_scalar(tid, base);
            tm_submit(tid);
            count++; tid++;

            new_task(tid, TASK_TYPE_VECTOR, (uint32_t)cur_blocks, DUR_SOFTMAX);
            Tensor cur_li_piece = view(tk->cur_li, base * 1024u, 0u,
                                       (uint32_t)(cur_blocks * 1024), 1u);
            Tensor cur_mi_piece = view(tk->cur_mi, base * 1024u, 0u,
                                       (uint32_t)(cur_blocks * 1024), 1u);
            Tensor exp_padded_piece = view(tk->exp_padded, base * 1024u, 0u,
                                           (uint32_t)(cur_blocks * 1024), 128u);
            tm_out(tid, cur_li_piece);
            tm_out(tid, cur_mi_piece);
            tm_out(tid, exp_padded_piece);
            tm_in(tid, row_piece);
            add_scalar(tid, 8);
            add_scalar(tid, 1024);
            add_scalar(tid, base);
            tm_submit(tid);
            count++; tid++;

            Tensor exp_piece = view(tk->exp_padded, base * 1024u, 0u,
                                    (uint32_t)(cur_blocks * 1024), 128u);
            new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_SV_MATMUL);
            Tensor oi_tmp_piece = view(tk->oi_tmp, base * 1024u, 0u,
                                       (uint32_t)(cur_blocks * 1024), 128u);
            tm_out(tid, oi_tmp_piece);
            tm_in_ro(tid, ext_block_table);
            tm_in(tid, exp_piece);
            tm_in(tid, tk->v_cache_update);
            add_scalar(tid, 8);
            add_scalar(tid, b * 32);
            add_scalar(tid, base);
            tm_submit(tid);
            count++; tid++;

            new_task(tid, TASK_TYPE_VECTOR, (uint32_t)cur_blocks, DUR_ONLINE_SOFTMAX);
            tm_in(tid, oi_tmp_piece);
            tm_in(tid, cur_mi_piece);
            tm_in(tid, cur_li_piece);
            Tensor attn_out_piece = view(g_attn_out[tile], (uint32_t)(b % 16),
                base * 1280u, 1u, cur_blocks * 1280u);
            tm_inout(tid, attn_out_piece);
            add_scalar(tid, 8);
            add_scalar(tid, base);
            tm_submit(tid);
            count++; tid++;
        }
    }
    return count;
}

static int desc_phase3(int tile) {
    int __pipe_tile = tile;
    TileTensors *t = &g_tiles[tile];
    int64_t b0 = tile * 16;
    int64_t cur_valid = (tile < 5) ? 16 : (90 - 80);
    uint32_t tid = p3_tid(tile, 0);
    int count = 0;

    for (int base = 0; base < 40; base += qwen3_blocks_per_task(40)) {
        int cur_blocks = qwen3_cur_blocks(40, base);
        new_task(tid, TASK_TYPE_MIX, (uint32_t)cur_blocks, DUR_OUT_PROJ);
        Tensor attn_out_tile = view(g_attn_out[tile], 0u, 0u, (uint32_t)cur_valid, 5120u);
        Tensor resid1_piece = view(t->resid1_tile, 0u, base * 128u, 16u,
                                   (uint32_t)(cur_blocks * 128));
        tm_in_ro(tid, ext_hidden_states);
        tm_in(tid, attn_out_tile);
        tm_in_ro(tid, ext_wo);
        tm_inout(tid, resid1_piece);
        tm_out(tid, t->gm_pipe_buffer_0);
        add_scalar(tid, b0);
        add_scalar(tid, cur_valid);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;
    }

    new_task(tid, TASK_TYPE_VECTOR, 1, DUR_POST_RMSNORM);
    tm_in(tid, t->resid1_tile);
    tm_out(tid, t->post_norm_tile);
    tm_in_ro(tid, ext_post_rms_weight);
    tm_submit(tid);
    count++; tid++;

    for (int base = 0; base < 34; base += qwen3_blocks_per_task(34)) {
        int cur_blocks = qwen3_cur_blocks(34, base);
        Tensor gate_piece = view(t->gate_tile, 0u, base * 512u, 16u,
                                  (uint32_t)(cur_blocks * 512));
        Tensor up_piece = view(t->up_tile, 0u, base * 512u, 16u,
                               (uint32_t)(cur_blocks * 512));

        new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_GATE_PROJ);
        tm_in(tid, t->post_norm_tile);
        tm_in_ro(tid, ext_w_gate);
        tm_inout(tid, gate_piece);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;

        new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_UP_PROJ);
        tm_in(tid, t->post_norm_tile);
        tm_in_ro(tid, ext_w_up);
        tm_inout(tid, up_piece);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;

        new_task(tid, TASK_TYPE_VECTOR, (uint32_t)cur_blocks, DUR_SILU);
        tm_in(tid, gate_piece);
        tm_in(tid, up_piece);
        Tensor mlp_piece = view(t->mlp_tile, 0u, base * 512u, 16u,
                                (uint32_t)(cur_blocks * 512));
        tm_inout(tid, mlp_piece);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;
    }

    for (int base = 0; base < 40; base += qwen3_blocks_per_task(40)) {
        int cur_blocks = qwen3_cur_blocks(40, base);
        Tensor down_piece = view(t->down_tile, 0u, base * 128u, 16u,
                                  (uint32_t)(cur_blocks * 128));
        Tensor resid1_piece = view(t->resid1_tile, 0u, base * 128u, 16u,
                                    (uint32_t)(cur_blocks * 128));

        new_task(tid, TASK_TYPE_CUBE, (uint32_t)cur_blocks, DUR_DOWN_PROJ);
        tm_in(tid, t->mlp_tile);
        tm_in_ro(tid, ext_w_down);
        tm_inout(tid, down_piece);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;

        new_task(tid, TASK_TYPE_VECTOR, (uint32_t)cur_blocks, DUR_DOWN_PROJ_RES);
        tm_in(tid, down_piece);
        tm_in(tid, resid1_piece);
        tm_out_ro(tid, ext_out);
        add_scalar(tid, cur_valid);
        add_scalar(tid, b0);
        add_scalar(tid, base);
        tm_submit(tid);
        count++; tid++;
    }
    return count;
}

static int tasks_in_phase(int tile, int phase) {
    if (phase == 0) return P1_PER_TILE;
    if (phase == 1) return g_tiles[tile].num_tokens * P2_PER_TOKEN;
    return P3_PER_TILE;
}

static inline int worker_tile_start(int worker_id) {
    return worker_id * NUM_ORCH_TILES / g_num_workers;
}
static inline int worker_tile_end(int worker_id) {
    return (worker_id + 1) * NUM_ORCH_TILES / g_num_workers;
}

static uint64_t g_pipe_orch_args;

static void *worker_thread_func(void *arg) {
    int worker_id = *(int *)arg;
    int ts = worker_tile_start(worker_id);
    int te = worker_tile_end(worker_id);
    pin_thread(96 + worker_id * 2);

    for (int t = ts; t < te; t++)
        tm_deps_init_pipe(t);

    atomic_fetch_add_explicit(&g_threads_at_barrier, 1, memory_order_seq_cst);
    while (atomic_load_explicit(&g_start_barrier, memory_order_acquire) == 0)
        pipe_spin();

    g_worker_stats[worker_id].s.start_ns = get_time_ns();
    if (worker_id == 0)
        g_pipeline_start_ns = g_worker_stats[worker_id].s.start_ns;

    int total_created = 0;
    for (int phase = 0; phase < 3; phase++) {
        for (int t = ts; t < te; t++) {
            int __pipe_tile = t;
            int created;
            if (phase == 0)      created = desc_phase1(t);
            else if (phase == 1) created = desc_phase2(t);
            else                 created = desc_phase3(t);
            total_created += created;
        }
    }

    g_worker_stats[worker_id].s.end_ns = get_time_ns();
    g_worker_stats[worker_id].s.task_count = (uint32_t)total_created;
    atomic_fetch_add_explicit(&g_submit_done_cnt, 1, memory_order_release);
    return NULL;
}

void aicpu_orchestration_entry(const uint64_t orch_args) {
    pthread_t worker_threads[NUM_ORCH_TILES];
    int worker_ids[NUM_ORCH_TILES];

    if (g_num_workers < 1) g_num_workers = 1;
    if (g_num_workers > NUM_ORCH_TILES) g_num_workers = NUM_ORCH_TILES;

    g_pipe_orch_args = orch_args;

    ext_hidden_states    = tensor_from_base_layout(orch_args + 0,  (uint32_t[]){90, 5120}, 2, BFLOAT16);
    ext_input_rms_weight = tensor_from_base_layout(orch_args + 1,  (uint32_t[]){1, 5120}, 2, FLOAT32);
    ext_wq               = tensor_from_base_layout(orch_args + 2,  (uint32_t[]){5120, 5120}, 2, BFLOAT16);
    ext_wk               = tensor_from_base_layout(orch_args + 3,  (uint32_t[]){5120, 1024}, 2, BFLOAT16);
    ext_wv               = tensor_from_base_layout(orch_args + 4,  (uint32_t[]){5120, 1024}, 2, BFLOAT16);
    ext_q_norm_weight    = tensor_from_base_layout(orch_args + 5,  (uint32_t[]){1, 128}, 2, FLOAT32);
    ext_k_norm_weight    = tensor_from_base_layout(orch_args + 6,  (uint32_t[]){1, 128}, 2, FLOAT32);
    ext_seq_lens         = tensor_from_base_layout(orch_args + 7,  (uint32_t[]){90}, 1, INT32);
    ext_block_table      = tensor_from_base_layout(orch_args + 8,  (uint32_t[]){2880}, 1, INT32);
    ext_slot_mapping     = tensor_from_base_layout(orch_args + 9,  (uint32_t[]){90}, 1, INT32);
    ext_rope_cos         = tensor_from_base_layout(orch_args + 10, (uint32_t[]){4096, 128}, 2, FLOAT32);
    ext_rope_sin         = tensor_from_base_layout(orch_args + 11, (uint32_t[]){4096, 128}, 2, FLOAT32);
    ext_k_cache          = tensor_from_base_layout(orch_args + 12, (uint32_t[]){2949120, 128}, 2, BFLOAT16);
    ext_v_cache          = tensor_from_base_layout(orch_args + 13, (uint32_t[]){2949120, 128}, 2, BFLOAT16);
    ext_wo               = tensor_from_base_layout(orch_args + 14, (uint32_t[]){5120, 5120}, 2, BFLOAT16);
    ext_post_rms_weight  = tensor_from_base_layout(orch_args + 15, (uint32_t[]){1, 5120}, 2, FLOAT32);
    ext_w_gate           = tensor_from_base_layout(orch_args + 16, (uint32_t[]){5120, 17408}, 2, BFLOAT16);
    ext_w_up             = tensor_from_base_layout(orch_args + 17, (uint32_t[]){5120, 17408}, 2, BFLOAT16);
    ext_w_down           = tensor_from_base_layout(orch_args + 18, (uint32_t[]){17408, 5120}, 2, BFLOAT16);
    ext_out              = tensor_from_base_layout(orch_args + 19, (uint32_t[]){90, 5120}, 2, BFLOAT16);
    (void)ext_seq_lens;
    (void)ext_slot_mapping;

    g_alloc_stat.s.start_ns = get_time_ns();

    g_q_proj     = alloc_tensors((uint32_t[2]){96, 5120}, 2, FLOAT32);
    g_k_proj     = alloc_tensors((uint32_t[2]){96, 1024}, 2, FLOAT32);
    g_v_proj     = alloc_tensors((uint32_t[2]){96, 1024}, 2, FLOAT32);
    g_q_proj_norm= alloc_tensors((uint32_t[2]){96, 5120}, 2, FLOAT32);
    g_k_proj_norm= alloc_tensors((uint32_t[2]){96, 1024}, 2, FLOAT32);
    for (int i = 0; i < NUM_ORCH_TILES; i++) {
        g_attn_out[i] = alloc_tensors((uint32_t[2]){16, 5120}, 2, BFLOAT16);
    }

    for (int t = 0; t < NUM_ORCH_TILES; t++) {
        g_tiles[t].num_tokens = (t < 5) ? 16 : 10;
        alloc_phase1_tensors(t);
        alloc_phase2_tokens(t);
        alloc_phase3_tensors(t);
    }

    g_alloc_stat.s.end_ns = get_time_ns();
    g_alloc_stat.s.task_count = 6 + NUM_ORCH_TILES;

    atomic_init(&g_start_barrier, 0);
    atomic_init(&g_threads_at_barrier, 0);
    atomic_init(&g_submit_done_cnt, 0);
    for (int t = 0; t < NUM_ORCH_TILES; t++) {
        worker_ids[t] = t;
        pred_ring_init_tile(t);
    }

    for (int w = 0; w < g_num_workers; w++) {
        pthread_create(&worker_threads[w], NULL, worker_thread_func, &worker_ids[w]);
    }

    while (atomic_load_explicit(&g_threads_at_barrier, memory_order_seq_cst)
           < g_num_workers)
        pipe_spin();

    atomic_store_explicit(&g_start_barrier, 1, memory_order_release);

    while (atomic_load_explicit(&g_submit_done_cnt, memory_order_acquire)
           < g_num_workers)
        pipe_spin();

    g_pipeline_end_ns = get_time_ns();

    for (int w = 0; w < g_num_workers; w++) {
        pthread_join(worker_threads[w], NULL);
    }

    for (int t = 0; t < NUM_ORCH_TILES; t++)
        g_subtask_cnt += g_subtask_cnt_per_tile[t].cnt;
    atomic_store(&g_task_id, TOTAL_TASKS);

    print_pipeline_stats(TOTAL_TASKS, (uint32_t)g_subtask_cnt);

#ifdef DAG_DUMP
    dump_dag(TOTAL_TASKS);
#endif

    for (int t = 0; t < NUM_ORCH_TILES; t++)
        pred_ring_deinit_tile(t);
}

#endif /* QWEN3_PIPELINE_H */
