/* Thin wrapper: includes only the desc case header and exposes a
 * callable function for use by orchestrator.c. */

#include "tensormap.h"
#include "log.h"
#include "qwen3_14b_decoder_desc.h"

int orc_desc_call(uint64_t orch_args, int thread_id, int *created_cnt)
{
    return orchestrator_desc(orch_args, thread_id, created_cnt);
}

/* ── Direct Tensor overlap check (no TmEntry conversion needed) ────────── */

static inline uint64_t tensor_extent_elem(const Tensor *t)
{
    if (t->is_contiguous) {
        uint64_t n = 1;
        for (uint32_t i = 0; i < t->ndims; i++)
            n *= t->shapes[i];
        return n;
    }
    return t->extent_elem_cache;
}

static inline TmOverlap tm_check_overlap_tensor(const Tensor *in, const Tensor *e)
{
    if (in->version > e->version)
        return TM_OVERLAP_OTHER;

    if (in->ndims == 2u && e->ndims == 2u) {
        uint64_t extent_elem;
        if (in->is_contiguous)
            extent_elem = (uint64_t)in->shapes[0] * in->shapes[1];
        else
            extent_elem = in->extent_elem_cache;

        uint64_t ent_extent;
        if (e->is_contiguous)
            ent_extent = (uint64_t)e->shapes[0] * e->shapes[1];
        else
            ent_extent = e->extent_elem_cache;

        const uint64_t in_end = in->start_offset + extent_elem;
        const uint64_t ent_end = e->start_offset + ent_extent;
        if (!(in_end > e->start_offset && ent_end > in->start_offset))
            return TM_OVERLAP_NONE;

        if (in->dtype != e->dtype)
            return TM_OVERLAP_OTHER;
        if (in->strides[0] != e->strides[0] || in->strides[1] != e->strides[1])
            return TM_OVERLAP_OTHER;
        if (e->strides[1] != 1u)
            return TM_OVERLAP_OTHER;
        if (e->strides[0] % e->strides[1] != 0u)
            return TM_OVERLAP_OTHER;

        const uint32_t ref_shape1 = e->strides[0] / e->strides[1];
        const uint32_t stride0 = e->strides[0];
        const uint64_t numel_storage =
            in->dtype != 0u ? in->buffer_size / (uint64_t)in->dtype : 0u;
        if (stride0 == 0u || numel_storage % stride0 != 0u)
            return TM_OVERLAP_OTHER;
        const uint32_t ref_shape0 = (uint32_t)(numel_storage / stride0);

        const uint32_t s0 = e->strides[0];
        const uint32_t s1 = e->strides[1];
        uint64_t in_remain = in->start_offset;
        uint64_t ent_remain = e->start_offset;
        const uint32_t in_off0 = (uint32_t)(in_remain / s0);
        in_remain %= s0;
        const uint32_t in_off1 = (uint32_t)(in_remain / s1);
        in_remain %= s1;
        const uint32_t ent_off0 = (uint32_t)(ent_remain / s0);
        ent_remain %= s0;
        const uint32_t ent_off1 = (uint32_t)(ent_remain / s1);
        ent_remain %= s1;
        if (in_remain != 0u || ent_remain != 0u)
            return TM_OVERLAP_OTHER;

        if ((uint64_t)in_off0 + in->shapes[0] > ref_shape0 ||
            (uint64_t)ent_off0 + e->shapes[0] > ref_shape0)
            return TM_OVERLAP_OTHER;
        if ((uint64_t)in_off1 + in->shapes[1] > ref_shape1 ||
            (uint64_t)ent_off1 + e->shapes[1] > ref_shape1)
            return TM_OVERLAP_OTHER;

        const uint64_t in_a1_0 = (uint64_t)in_off0 + in->shapes[0];
        const uint64_t ent_b1_0 = (uint64_t)ent_off0 + e->shapes[0];
        if (!(in_a1_0 > ent_off0 && ent_b1_0 > in_off0))
            return TM_OVERLAP_NONE;
        bool input_contains_entry = (in_off0 <= ent_off0 && ent_b1_0 <= in_a1_0);

        const uint64_t in_a1_1 = (uint64_t)in_off1 + in->shapes[1];
        const uint64_t ent_b1_1 = (uint64_t)ent_off1 + e->shapes[1];
        if (!(in_a1_1 > ent_off1 && ent_b1_1 > in_off1))
            return TM_OVERLAP_NONE;
        if (!(in_off1 <= ent_off1 && ent_b1_1 <= in_a1_1))
            input_contains_entry = false;

        return input_contains_entry ? TM_OVERLAP_COVERED : TM_OVERLAP_OTHER;
    }

    uint64_t extent_elem;
    if (in->is_contiguous) {
        extent_elem = 1;
        for (uint32_t i = 0; i < in->ndims; i++)
            extent_elem *= in->shapes[i];
    } else {
        extent_elem = in->extent_elem_cache;
    }

    const uint64_t in_begin = in->start_offset;
    const uint64_t in_end = in->start_offset + extent_elem;
    const uint64_t ent_begin = e->start_offset;
    const uint64_t ent_end = e->start_offset + tensor_extent_elem(e);
    if (!(in_end > ent_begin && ent_end > in_begin))
        return TM_OVERLAP_NONE;

    if (in->dtype != e->dtype || in->ndims != e->ndims || in->ndims == 0)
        return TM_OVERLAP_OTHER;
    for (uint32_t i = 0; i < in->ndims; i++) {
        if (in->strides[i] != e->strides[i])
            return TM_OVERLAP_OTHER;
    }
    if (e->strides[in->ndims - 1u] != 1u)
        return TM_OVERLAP_OTHER;
    for (uint32_t i = 1; i < in->ndims; i++) {
        if (e->strides[i - 1u] % e->strides[i] != 0u)
            return TM_OVERLAP_OTHER;
    }

    uint32_t ref_shapes[TM_MAX_DIMS] = {0};
    for (uint32_t i = 1; i < in->ndims; i++)
        ref_shapes[i] = e->strides[i - 1u] / e->strides[i];
    const uint32_t stride0 = e->strides[0];
    const uint64_t numel_storage =
        in->dtype != 0u ? in->buffer_size / (uint64_t)in->dtype : 0u;
    if (stride0 == 0u || numel_storage % stride0 != 0u)
        return TM_OVERLAP_OTHER;
    ref_shapes[0] = (uint32_t)(numel_storage / stride0);

    uint32_t in_offsets[TM_MAX_DIMS] = {0};
    uint32_t ent_offsets[TM_MAX_DIMS] = {0};
    uint64_t in_remain = in->start_offset;
    uint64_t ent_remain = e->start_offset;
    for (uint32_t i = 0; i < in->ndims; i++) {
        const uint32_t s = e->strides[i];
        in_offsets[i] = (uint32_t)(in_remain / s);
        ent_offsets[i] = (uint32_t)(ent_remain / s);
        in_remain %= s;
        ent_remain %= s;
    }
    if (in_remain != 0u || ent_remain != 0u)
        return TM_OVERLAP_OTHER;

    for (uint32_t i = 0; i < in->ndims; i++) {
        if ((uint64_t)in_offsets[i] + in->shapes[i] > ref_shapes[i] ||
            (uint64_t)ent_offsets[i] + e->shapes[i] > ref_shapes[i])
            return TM_OVERLAP_OTHER;
    }

    bool input_contains_entry = true;
    for (uint32_t i = 0; i < in->ndims; i++) {
        const uint64_t a0 = in_offsets[i];
        const uint64_t a1 = a0 + in->shapes[i];
        const uint64_t b0 = ent_offsets[i];
        const uint64_t b1 = b0 + e->shapes[i];
        if (!(a1 > b0 && b1 > a0))
            return TM_OVERLAP_NONE;
        if (!(a0 <= b0 && b1 <= a1))
            input_contains_entry = false;
    }
    return input_contains_entry ? TM_OVERLAP_COVERED : TM_OVERLAP_OTHER;
}

/* ── Merged submit: direct scan g_task_tensor_buf ──────────────────────── */

#define MAX_PREDS_PER_TASK 1024
#define MAX_TASKS 4096

static uint32_t *g_pred_buf;
static int g_total_tasks;

void orc_submit_init(int total_tasks)
{
    g_total_tasks = total_tasks;
    g_pred_buf = calloc((size_t)total_tasks * MAX_PREDS_PER_TASK, sizeof(uint32_t));
}

void orc_submit_task(uint32_t task_id)
{
    struct task_tensor_desc *desc = &g_task_tensor_buf[task_id & RING_MASK];
    uint32_t *preds = &g_pred_buf[(size_t)task_id * MAX_PREDS_PER_TASK];
    int pn = 0;
    uint64_t seen[MAX_TASKS / 64];
    memset(seen, 0, sizeof seen);

    for (int pass = 0; pass < 2; pass++) {
        uint16_t cnt = (pass == 0) ? desc->in_cnt : desc->inout_cnt;
        Tensor *tensors = (pass == 0) ? desc->in_data : desc->inout_data;

        for (uint16_t i = 0; i < cnt; i++) {
            const Tensor *in = &tensors[i];
            for (uint32_t prod = 0; prod < task_id; prod++) {
                if (seen[prod >> 6] & ((uint64_t)1 << (prod & 63)))
                    continue;
                struct task_tensor_desc *pdesc = &g_task_tensor_buf[prod & RING_MASK];
                bool found = false;
                for (uint16_t j = 0; j < pdesc->out_cnt && !found; j++) {
                    if (pdesc->out_data[j].buffer_addr != in->buffer_addr)
                        continue;
                    if (tm_check_overlap_tensor(in, &pdesc->out_data[j]) != TM_OVERLAP_NONE)
                        found = true;
                }
                for (uint16_t j = 0; j < pdesc->inout_cnt && !found; j++) {
                    if (pdesc->inout_data[j].buffer_addr != in->buffer_addr)
                        continue;
                    if (tm_check_overlap_tensor(in, &pdesc->inout_data[j]) != TM_OVERLAP_NONE)
                        found = true;
                }
                if (found) {
                    seen[prod >> 6] |= (uint64_t)1 << (prod & 63);
                    if (pn < MAX_PREDS_PER_TASK)
                        preds[pn++] = prod;
                }
            }
        }
    }

    g_predecessors[task_id].exp = preds;
    g_predecessors[task_id].cnt = (uint32_t)pn;
}
