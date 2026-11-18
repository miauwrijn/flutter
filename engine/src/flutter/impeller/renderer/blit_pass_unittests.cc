// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdint>
#include "flutter/display_list/dl_builder.h"
#include "flutter/impeller/display_list/aiks_unittests.h"
#include "flutter/impeller/display_list/dl_image_impeller.h"
#include "fml/mapping.h"
#include "fml/synchronization/waitable_event.h"
#include "gtest/gtest.h"
#include "impeller/base/validation.h"
#include "impeller/core/device_buffer.h"
#include "impeller/core/device_buffer_descriptor.h"
#include "impeller/core/formats.h"
#include "impeller/core/texture_descriptor.h"
#include "impeller/playground/playground_test.h"
#include "impeller/renderer/command_buffer.h"

namespace impeller {
namespace testing {

using flutter::DisplayListBuilder;
using flutter::DlColor;
using flutter::DlImageSampling;
using flutter::DlPaint;

using BlitPassTest = AiksTest;
INSTANTIATE_PLAYGROUND_SUITE(BlitPassTest);

TEST_P(BlitPassTest, BlitAcrossDifferentPixelFormatsFails) {
  ScopedValidationDisable scope;  // avoid noise in output.
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor src_desc;
  src_desc.format = PixelFormat::kA8UNormInt;
  src_desc.size = {100, 100};
  src_desc.storage_mode = StorageMode::kHostVisible;
  auto src = context->GetResourceAllocator()->CreateTexture(src_desc);

  TextureDescriptor dst_format;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {100, 100};
  dst_format.storage_mode = StorageMode::kHostVisible;
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  EXPECT_FALSE(blit_pass->AddCopy(src, dst));
}

TEST_P(BlitPassTest, BlitAcrossDifferentSampleCountsFails) {
  ScopedValidationDisable scope;  // avoid noise in output.
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor src_desc;
  src_desc.format = PixelFormat::kR8G8B8A8UNormInt;
  src_desc.sample_count = SampleCount::kCount4;
  src_desc.size = {100, 100};
  auto src = context->GetResourceAllocator()->CreateTexture(src_desc);

  TextureDescriptor dst_format;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {100, 100};
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  EXPECT_FALSE(blit_pass->AddCopy(src, dst));
}

TEST_P(BlitPassTest, BlitPassesForMatchingFormats) {
  ScopedValidationDisable scope;  // avoid noise in output.
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor src_desc;
  src_desc.format = PixelFormat::kR8G8B8A8UNormInt;
  src_desc.size = {100, 100};
  src_desc.storage_mode = StorageMode::kHostVisible;
  auto src = context->GetResourceAllocator()->CreateTexture(src_desc);

  TextureDescriptor dst_format;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {100, 100};
  dst_format.storage_mode = StorageMode::kHostVisible;
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  EXPECT_TRUE(blit_pass->AddCopy(src, dst));
}

TEST_P(BlitPassTest, ChecksInvalidSliceParameters) {
  ScopedValidationDisable scope;  // avoid noise in output.
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor dst_format;
  dst_format.storage_mode = StorageMode::kDevicePrivate;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {100, 100};
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  DeviceBufferDescriptor src_format;
  src_format.size = 40000;
  src_format.storage_mode = StorageMode::kHostVisible;
  auto src = context->GetResourceAllocator()->CreateBuffer(src_format);

  ASSERT_TRUE(dst);
  ASSERT_TRUE(src);

  EXPECT_FALSE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                  std::nullopt, "", /*mip_level=*/0,
                                  /*slice=*/25));
  EXPECT_FALSE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                  std::nullopt, "", /*mip_level=*/0,
                                  /*slice=*/6));
  EXPECT_TRUE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                 std::nullopt, "", /*mip_level=*/0,
                                 /*slice=*/0));
}

TEST_P(BlitPassTest, CanBlitSmallRegionToUninitializedTexture) {
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor dst_format;
  dst_format.storage_mode = StorageMode::kDevicePrivate;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {1000, 1000};
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  DeviceBufferDescriptor src_format;
  src_format.size = 4;
  src_format.storage_mode = StorageMode::kHostVisible;
  auto src = context->GetResourceAllocator()->CreateBuffer(src_format);

  ASSERT_TRUE(dst);

  EXPECT_TRUE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                 IRect::MakeLTRB(0, 0, 1, 1), "",
                                 /*mip_level=*/0, /*slice=*/0));
  EXPECT_TRUE(blit_pass->EncodeCommands());
  EXPECT_TRUE(context->GetCommandQueue()->Submit({std::move(cmd_buffer)}).ok());
}

TEST_P(BlitPassTest, ChecksInvalidMipLevelParameter) {
  ScopedValidationDisable scope;
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor dst_format;
  dst_format.storage_mode = StorageMode::kDevicePrivate;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {1000, 1000};
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  DeviceBufferDescriptor src_format;
  src_format.size = 4;
  src_format.storage_mode = StorageMode::kHostVisible;
  auto src = context->GetResourceAllocator()->CreateBuffer(src_format);

  ASSERT_TRUE(dst);

  EXPECT_FALSE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                  IRect::MakeLTRB(0, 0, 1, 1), "",
                                  /*mip_level=*/1, /*slice=*/0));

  EXPECT_TRUE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                 IRect::MakeLTRB(0, 0, 1, 1), "",
                                 /*mip_level=*/0, /*slice=*/0));
}

TEST_P(BlitPassTest, CanBlitToHigherTextureMipLevels) {
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor dst_format;
  dst_format.storage_mode = StorageMode::kDevicePrivate;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {1000, 1000};
  dst_format.mip_count = 4;
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  DeviceBufferDescriptor src_format;
  src_format.size = 4;
  src_format.storage_mode = StorageMode::kHostVisible;
  auto src = context->GetResourceAllocator()->CreateBuffer(src_format);

  ASSERT_TRUE(dst);

  EXPECT_TRUE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(src), dst,
                                 IRect::MakeLTRB(0, 0, 1, 1), "",
                                 /*mip_level=*/1, /*slice=*/0));
  EXPECT_TRUE(blit_pass->EncodeCommands());
  EXPECT_TRUE(context->GetCommandQueue()->Submit({std::move(cmd_buffer)}).ok());
}

TEST_P(BlitPassTest, CanResizeTextures) {
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  TextureDescriptor dst_format;
  dst_format.storage_mode = StorageMode::kDevicePrivate;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {10, 10};
  dst_format.usage = TextureUsage::kShaderRead | TextureUsage::kShaderWrite;
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  TextureDescriptor src_format;
  src_format.storage_mode = StorageMode::kDevicePrivate;
  src_format.format = PixelFormat::kR8G8B8A8UNormInt;
  src_format.size = {100, 100};
  auto src = context->GetResourceAllocator()->CreateTexture(src_format);

  std::vector<uint8_t> bytes(src_format.GetByteSizeOfBaseMipLevel());
  for (auto i = 0u; i < src_format.GetByteSizeOfBaseMipLevel(); i += 4) {
    // RGBA
    bytes[i + 0] = 255;
    bytes[i + 1] = 0;
    bytes[i + 2] = 0;
    bytes[i + 3] = 255;
  }
  auto mapping = fml::DataMapping(bytes);
  auto staging = context->GetResourceAllocator()->CreateBufferWithCopy(mapping);

  ASSERT_TRUE(dst);
  ASSERT_TRUE(src);
  ASSERT_TRUE(staging);

  EXPECT_TRUE(blit_pass->AddCopy(DeviceBuffer::AsBufferView(staging), src));
  EXPECT_TRUE(blit_pass->ResizeTexture(src, dst));
  EXPECT_TRUE(blit_pass->EncodeCommands());
  EXPECT_TRUE(context->GetCommandQueue()->Submit({std::move(cmd_buffer)}).ok());
}

TEST_P(BlitPassTest, CanResizeTexturesPlayground) {
  auto context = GetContext();
  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  std::shared_ptr<Texture> src = CreateTextureForFixture("kalimba.jpg");

  TextureDescriptor dst_format;
  dst_format.storage_mode = StorageMode::kDevicePrivate;
  dst_format.format = PixelFormat::kR8G8B8A8UNormInt;
  dst_format.size = {src->GetSize().width / 2, src->GetSize().height};
  dst_format.usage = TextureUsage::kShaderRead | TextureUsage::kShaderWrite;
  auto dst = context->GetResourceAllocator()->CreateTexture(dst_format);

  ASSERT_TRUE(dst);
  ASSERT_TRUE(src);

  EXPECT_TRUE(blit_pass->ResizeTexture(src, dst));
  EXPECT_TRUE(blit_pass->EncodeCommands());
  EXPECT_TRUE(context->GetCommandQueue()->Submit({std::move(cmd_buffer)}).ok());

  DisplayListBuilder builder;
  builder.Scale(GetContentScale().x, GetContentScale().y);
  DlPaint paint;
  paint.setColor(DlColor::kRed());
  auto image = DlImageImpeller::Make(dst);
  builder.DrawImage(image, flutter::DlPoint(100.0, 100.0),
                    DlImageSampling::kNearestNeighbor, &paint);
  ASSERT_TRUE(OpenPlaygroundHere(builder.Build()));
}

namespace {

std::shared_ptr<DeviceBuffer> CreateBuffer(
    const std::shared_ptr<Context>& context,
    StorageMode storage_mode,
    size_t size,
    bool readback = false) {
  DeviceBufferDescriptor desc;
  desc.storage_mode = storage_mode;
  desc.size = size;
  desc.readback = readback;
  return context->GetResourceAllocator()->CreateBuffer(desc);
}

}  // namespace

TEST_P(BlitPassTest, CanCopyBufferToBufferForReadback) {
  auto context = GetContext();

  constexpr size_t kSize = 256;
  std::vector<uint8_t> bytes(kSize);
  for (size_t i = 0; i < kSize; i++) {
    bytes[i] = static_cast<uint8_t>(i);
  }
  auto source = context->GetResourceAllocator()->CreateBufferWithCopy(
      fml::DataMapping(bytes));
  // Copy through a device private buffer, so the bytes only reach the host
  // through the copies, and through two copies in one pass, so the second
  // copy has to be ordered after the first.
  auto device_private =
      CreateBuffer(context, StorageMode::kDevicePrivate, kSize);
  auto readback = CreateBuffer(context, StorageMode::kHostVisible,
                               /*size=*/kSize, /*readback=*/true);
  ASSERT_TRUE(source);
  ASSERT_TRUE(device_private);
  ASSERT_TRUE(readback);

  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();
  // Copy the second half of the source to the start of the device private
  // buffer and the first half after it, to check offsets.
  EXPECT_TRUE(blit_pass->AddCopy(
      BufferView(source, Range{kSize / 2, kSize / 2}), device_private,
      /*destination_offset=*/0));
  EXPECT_TRUE(blit_pass->AddCopy(BufferView(source, Range{0, kSize / 2}),
                                 device_private,
                                 /*destination_offset=*/kSize / 2));
  EXPECT_TRUE(blit_pass->AddCopy(BufferView(device_private, Range{0, kSize}),
                                 readback));
  EXPECT_TRUE(blit_pass->EncodeCommands());

  fml::AutoResetWaitableEvent latch;
  CommandBuffer::Status result = CommandBuffer::Status::kPending;
  ASSERT_TRUE(context->GetCommandQueue()
                  ->Submit({cmd_buffer},
                           [&latch, &result](CommandBuffer::Status status) {
                             result = status;
                             latch.Signal();
                           })
                  .ok());
  latch.Wait();
  ASSERT_EQ(result, CommandBuffer::Status::kCompleted);

  readback->Invalidate();
  const uint8_t* contents = readback->OnGetContents();
  ASSERT_NE(contents, nullptr);
  for (size_t i = 0; i < kSize; i++) {
    ASSERT_EQ(contents[i], static_cast<uint8_t>((i + kSize / 2) % kSize))
        << "at byte " << i;
  }
}

TEST_P(BlitPassTest, BufferToBufferCopyChecksBounds) {
  auto context = GetContext();
  auto source = CreateBuffer(context, StorageMode::kHostVisible, 64);
  auto destination = CreateBuffer(context, StorageMode::kHostVisible, 32);
  ASSERT_TRUE(source);
  ASSERT_TRUE(destination);

  auto cmd_buffer = context->CreateCommandBuffer();
  auto blit_pass = cmd_buffer->CreateBlitPass();

  ScopedValidationDisable scoped;
  // Reads past the end of the source.
  EXPECT_FALSE(
      blit_pass->AddCopy(BufferView(source, Range{48, 32}), destination));
  // Writes past the end of the destination.
  EXPECT_FALSE(blit_pass->AddCopy(BufferView(source, Range{0, 32}), destination,
                                  /*destination_offset=*/16));
  // Overlapping ranges of the same buffer.
  EXPECT_FALSE(blit_pass->AddCopy(BufferView(source, Range{0, 32}), source,
                                  /*destination_offset=*/16));
  // No destination.
  EXPECT_FALSE(blit_pass->AddCopy(BufferView(source, Range{0, 32}),
                                  std::shared_ptr<DeviceBuffer>()));

  // In bounds, including disjoint ranges of one buffer and an empty copy.
  EXPECT_TRUE(
      blit_pass->AddCopy(BufferView(source, Range{0, 32}), destination));
  EXPECT_TRUE(blit_pass->AddCopy(BufferView(source, Range{0, 32}), source,
                                 /*destination_offset=*/32));
  EXPECT_TRUE(
      blit_pass->AddCopy(BufferView(source, Range{64, 0}), destination));
}

}  // namespace testing
}  // namespace impeller
