/*
 * spin.h - Atomic spin utilities
 *
 * Low-level atomic operations for spin waiting and synchronization.
 * C11 standard with stdatomic.
 */

#ifndef SPIN_H
#define SPIN_H

#include <stdatomic.h>

#if defined(__x86_64__) || defined(__i386__)
#define CPU_RELAX() __asm__ __volatile__("pause" ::: "memory")
#elif defined(__aarch64__) || defined(__arm__)
#define CPU_RELAX() __asm__ __volatile__("nop" ::: "memory")
#else
#define CPU_RELAX() atomic_thread_fence(memory_order_seq_cst)
#endif

/*
 * Memory barrier for spin-wait loops
 */
static inline void spin_wait(void)
{
    CPU_RELAX();
}

#endif /* SPIN_H */