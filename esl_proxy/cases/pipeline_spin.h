#ifndef PIPELINE_SPIN_H
#define PIPELINE_SPIN_H

#include <stdatomic.h>
#include <stdint.h>

#if defined(__aarch64__)
#define PIPE_PAUSE() __asm__ volatile("yield")
#elif defined(__x86_64__)
#define PIPE_PAUSE() __asm__ volatile("pause")
#else
#define PIPE_PAUSE() ((void)0)
#endif

static inline void pipe_spin(void){
    __asm__ volatile("nop");
}

#endif /* PIPELINE_SPIN_H */
