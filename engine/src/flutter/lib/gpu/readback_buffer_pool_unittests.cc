// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "flutter/lib/gpu/readback_buffer_pool.h"

#include <memory>

#include "flutter/testing/testing.h"  // IWYU pragma: keep
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "impeller/renderer/testing/mocks.h"

namespace flutter {
namespace gpu {
namespace {

using ::testing::_;
using ::testing::NiceMock;

// An allocator that hands out mock buffers and counts the allocations.
class CountingAllocator {
 public:
  CountingAllocator() {
    ON_CALL(allocator_, OnCreateBuffer(_))
        .WillByDefault([this](const impeller::DeviceBufferDescriptor& desc) {
          allocations_++;
          last_descriptor_ = desc;
          return std::make_shared<
              NiceMock<impeller::testing::MockDeviceBuffer>>(desc);
        });
  }

  impeller::Allocator& Get() { return allocator_; }
  int GetAllocations() const { return allocations_; }
  const impeller::DeviceBufferDescriptor& GetLastDescriptor() const {
    return last_descriptor_;
  }

 private:
  NiceMock<impeller::testing::MockAllocator> allocator_;
  int allocations_ = 0;
  impeller::DeviceBufferDescriptor last_descriptor_;
};

TEST(ReadbackBufferPoolTest, BucketsArePowersOfTwoWithAMinimum) {
  EXPECT_EQ(ReadbackBufferPool::BucketSize(0), 256u);
  EXPECT_EQ(ReadbackBufferPool::BucketSize(1), 256u);
  EXPECT_EQ(ReadbackBufferPool::BucketSize(256), 256u);
  EXPECT_EQ(ReadbackBufferPool::BucketSize(257), 512u);
  EXPECT_EQ(ReadbackBufferPool::BucketSize(4000), 4096u);
}

TEST(ReadbackBufferPoolTest, AllocatesHostVisibleReadbackBuffers) {
  CountingAllocator allocator;
  ReadbackBufferPool pool;
  auto buffer = pool.Acquire(allocator.Get(), 100);
  ASSERT_TRUE(buffer);
  EXPECT_EQ(allocator.GetAllocations(), 1);
  EXPECT_EQ(allocator.GetLastDescriptor().storage_mode,
            impeller::StorageMode::kHostVisible);
  EXPECT_TRUE(allocator.GetLastDescriptor().readback);
  EXPECT_EQ(allocator.GetLastDescriptor().size, 256u);
}

TEST(ReadbackBufferPoolTest, ReusesReleasedBuffersOfTheSameBucket) {
  CountingAllocator allocator;
  ReadbackBufferPool pool;
  auto first = pool.Acquire(allocator.Get(), 1000);
  impeller::DeviceBuffer* first_address = first.get();
  pool.Release(std::move(first));
  EXPECT_EQ(pool.GetPooledBufferCount(), 1u);
  EXPECT_EQ(pool.GetPooledBytes(), 1024u);

  // A read of another size in the same bucket gets the released buffer.
  auto second = pool.Acquire(allocator.Get(), 600);
  EXPECT_EQ(second.get(), first_address);
  EXPECT_EQ(allocator.GetAllocations(), 1);
  EXPECT_EQ(pool.GetPooledBufferCount(), 0u);

  // A read in another bucket allocates.
  auto third = pool.Acquire(allocator.Get(), 100);
  EXPECT_NE(third.get(), first_address);
  EXPECT_EQ(allocator.GetAllocations(), 2);
}

TEST(ReadbackBufferPoolTest, ConcurrentReadsGetDistinctBuffers) {
  CountingAllocator allocator;
  ReadbackBufferPool pool;
  auto first = pool.Acquire(allocator.Get(), 512);
  auto second = pool.Acquire(allocator.Get(), 512);
  EXPECT_NE(first.get(), second.get());
  EXPECT_EQ(allocator.GetAllocations(), 2);
}

TEST(ReadbackBufferPoolTest, KeepsAtMostTheByteAndBucketLimits) {
  CountingAllocator allocator;
  ReadbackBufferPool pool(/*max_pooled_bytes=*/2048);
  std::vector<std::shared_ptr<impeller::DeviceBuffer>> buffers;
  for (int i = 0; i < 4; i++) {
    buffers.push_back(pool.Acquire(allocator.Get(), 1024));
  }
  for (auto& buffer : buffers) {
    pool.Release(std::move(buffer));
  }
  // Two 1024-byte buffers fill the 2048-byte budget; the rest are freed.
  EXPECT_EQ(pool.GetPooledBufferCount(), 2u);
  EXPECT_EQ(pool.GetPooledBytes(), 2048u);

  ReadbackBufferPool roomy_pool;
  buffers.clear();
  for (size_t i = 0; i < ReadbackBufferPool::kMaxBuffersPerBucket + 2; i++) {
    buffers.push_back(roomy_pool.Acquire(allocator.Get(), 256));
  }
  for (auto& buffer : buffers) {
    roomy_pool.Release(std::move(buffer));
  }
  EXPECT_EQ(roomy_pool.GetPooledBufferCount(),
            ReadbackBufferPool::kMaxBuffersPerBucket);
}

TEST(ReadbackBufferPoolTest, IgnoresNullBuffers) {
  ReadbackBufferPool pool;
  pool.Release(nullptr);
  EXPECT_EQ(pool.GetPooledBufferCount(), 0u);
}

}  // namespace
}  // namespace gpu
}  // namespace flutter
