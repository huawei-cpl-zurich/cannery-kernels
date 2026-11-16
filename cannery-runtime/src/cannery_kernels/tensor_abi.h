#ifndef CANNERY_TENSOR_ABI_V2_H
#define CANNERY_TENSOR_ABI_V2_H
#include <stdint.h>

/* All tensors are contiguous; sizes/strides are element counts. Unused slots
 * are zero. dtype: f32=1, f16=2, bf16=3, i8=4, i32=5, i64=6, bool=7.
 * Storage is borrowed and must remain alive until the stream completes.
 * Producers must enforce the declared mutation policy and static signature. */
typedef struct {
  void *data;
  uint64_t sizes[4];
  uint64_t strides[4];
  uint32_t rank;
  uint32_t dtype;
} CanneryTensorV2;

/* kind: bool=1, signed i64=2, finite IEEE f64=3. reserved must be zero.
 * bits stores the value's bit pattern, not a pointer to host memory. */
typedef struct {
  uint32_t kind;
  uint32_t reserved;
  uint64_t bits;
} CanneryScalarV2;

#ifdef __cplusplus
extern "C" {
#endif
/* Return zero on successful enqueue, a nonzero native error otherwise.
 * No ownership transfer, synchronization, allocation, or Python/JIT execution.
 * Callers still synchronize the stream to observe asynchronous failures.
 * Counts: 1..32 inputs/outputs, 0..32 scalars; aggregate tensors <=32 MiB. */
int32_t cannery_launch_tensor_v2(void *stream,
    uint32_t ninputs, const CanneryTensorV2 *inputs,
    uint32_t noutputs, const CanneryTensorV2 *outputs,
    uint32_t nscalars, const CanneryScalarV2 *scalars);
#ifdef __cplusplus
}
#endif
#endif
