// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "flutter/lib/gpu/readback_buffer_pool.h"

#include <utility>

#include "impeller/core/device_buffer_descriptor.h"

namespace flutter {
namespace gpu {

ReadbackBufferPool::ReadbackBufferPool(size_t max_pooled_bytes)
    : max_pooled_bytes_(max_pooled_bytes) {}

ReadbackBufferPool::~ReadbackBufferPool() = default;

size_t ReadbackBufferPool::BucketSize(size_t length) {
  size_t size = kMinimumSize;
  while (size < length) {
    size *= 2u;
  }
  return size;
}

std::shared_ptr<impeller::DeviceBuffer> ReadbackBufferPool::Acquire(
    impeller::Allocator& allocator,
    size_t length) {
  const size_t size = BucketSize(length);
  {
    impeller::Lock lock(mutex_);
    auto found = free_buffers_.find(size);
    if (found != free_buffers_.end() && !found->second.empty()) {
      std::shared_ptr<impeller::DeviceBuffer> buffer =
          std::move(found->second.back());
      found->second.pop_back();
      pooled_bytes_ -= size;
      return buffer;
    }
  }
  impeller::DeviceBufferDescriptor desc;
  desc.storage_mode = impeller::StorageMode::kHostVisible;
  desc.size = size;
  desc.readback = true;
  std::shared_ptr<impeller::DeviceBuffer> buffer = allocator.CreateBuffer(desc);
  if (buffer) {
    buffer->SetLabel("Flutter GPU readback");
  }
  return buffer;
}

void ReadbackBufferPool::Release(
    std::shared_ptr<impeller::DeviceBuffer> buffer) {
  if (!buffer) {
    return;
  }
  const size_t size = buffer->GetDeviceBufferDescriptor().size;
  impeller::Lock lock(mutex_);
  std::vector<std::shared_ptr<impeller::DeviceBuffer>>& bucket =
      free_buffers_[size];
  if (pooled_bytes_ + size > max_pooled_bytes_ ||
      bucket.size() >= kMaxBuffersPerBucket) {
    // Freed when `buffer` goes out of scope.
    return;
  }
  bucket.push_back(std::move(buffer));
  pooled_bytes_ += size;
}

size_t ReadbackBufferPool::GetPooledBytes() const {
  impeller::Lock lock(mutex_);
  return pooled_bytes_;
}

size_t ReadbackBufferPool::GetPooledBufferCount() const {
  impeller::Lock lock(mutex_);
  size_t count = 0u;
  for (const auto& [size, buffers] : free_buffers_) {
    count += buffers.size();
  }
  return count;
}

}  // namespace gpu
}  // namespace flutter
