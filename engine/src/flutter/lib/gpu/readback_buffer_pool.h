// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef FLUTTER_LIB_GPU_READBACK_BUFFER_POOL_H_
#define FLUTTER_LIB_GPU_READBACK_BUFFER_POOL_H_

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "flutter/fml/macros.h"
#include "impeller/base/thread.h"
#include "impeller/core/allocator.h"
#include "impeller/core/device_buffer.h"

namespace flutter {
namespace gpu {

/// Host-visible staging buffers for `CommandBuffer.readBuffer` and
/// `readTexture`, reused across reads instead of allocated for each one.
///
/// Buffers are sized in power-of-two buckets of at least `kMinimumSize`
/// bytes, so a read gets a buffer at least as large as it needs. A buffer is
/// returned with `Release` once its command buffer completed and its bytes
/// were copied out, so a pooled buffer is never in use by the GPU. The pool
/// keeps at most `max_pooled_bytes` in total and `kMaxBuffersPerBucket`
/// buffers of each size; buffers beyond that are freed.
class ReadbackBufferPool {
 public:
  static constexpr size_t kMinimumSize = 256u;
  static constexpr size_t kMaxBuffersPerBucket = 4u;
  static constexpr size_t kDefaultMaxPooledBytes = 8u * 1024u * 1024u;

  explicit ReadbackBufferPool(
      size_t max_pooled_bytes = kDefaultMaxPooledBytes);

  ~ReadbackBufferPool();

  /// A readback buffer of at least [length] bytes: a pooled one if one of
  /// its size is free, or a new one from [allocator]. Null if allocation
  /// fails.
  std::shared_ptr<impeller::DeviceBuffer> Acquire(
      impeller::Allocator& allocator,
      size_t length);

  /// Returns [buffer], which must come from `Acquire` and no longer be in
  /// use, to the pool, or frees it if the pool is full.
  void Release(std::shared_ptr<impeller::DeviceBuffer> buffer);

  /// The bucket size a read of [length] bytes is served from.
  static size_t BucketSize(size_t length);

  size_t GetPooledBytes() const;

  size_t GetPooledBufferCount() const;

 private:
  const size_t max_pooled_bytes_;
  mutable impeller::Mutex mutex_;
  std::map<size_t, std::vector<std::shared_ptr<impeller::DeviceBuffer>>>
      free_buffers_ IPLR_GUARDED_BY(mutex_);
  size_t pooled_bytes_ IPLR_GUARDED_BY(mutex_) = 0u;

  FML_DISALLOW_COPY_AND_ASSIGN(ReadbackBufferPool);
};

}  // namespace gpu
}  // namespace flutter

#endif  // FLUTTER_LIB_GPU_READBACK_BUFFER_POOL_H_
