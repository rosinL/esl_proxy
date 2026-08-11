/* Thin wrapper: includes only the alloc case header and exposes a
 * callable function for use by orchestrator.c. */

#include "log.h"
#include "qwen3_14b_decoder_alloc.h"

void orc_alloc_call(uint64_t orch_args)
{
    uintptr_t fake_base = 0x10000000ULL;
    size_t fake_size = 0x80000000ULL;
    g_mem_pool.base = (void *)fake_base;
    g_mem_pool.size = fake_size;
    atomic_store(&g_mem_pool.tail, fake_base);
    atomic_store(&g_mem_pool.head, fake_base);

    orchestrator_alloc(orch_args);
}
