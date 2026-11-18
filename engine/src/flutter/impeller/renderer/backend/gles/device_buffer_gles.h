// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_DEVICE_BUFFER_GLES_H_
#define FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_DEVICE_BUFFER_GLES_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include "impeller/base/allocation.h"
#include "impeller/base/backend_cast.h"
#include "impeller/base/thread.h"
#include "impeller/core/device_buffer.h"
#include "impeller/renderer/backend/gles/reactor_gles.h"

namespace impeller {

class DeviceBufferGLES final
    : public DeviceBuffer,
      public BackendCast<DeviceBufferGLES, DeviceBuffer> {
 public:
  DeviceBufferGLES(DeviceBufferDescriptor desc,
                   std::shared_ptr<ReactorGLES> reactor,
                   std::unique_ptr<Allocation> backing_store);

  // |DeviceBuffer|
  ~DeviceBufferGLES() override;

  /// The host-side shadow copy of the buffer. If the GPU wrote the buffer,
  /// the shadow copy is synchronized from the GPU first, so this must be
  /// called on the reactor.
  const uint8_t* GetBufferData() const;

  void UpdateBufferData(
      const std::function<void(uint8_t*, size_t length)>& update_buffer_data);

  /// Copies [source_range] of [source] into the shadow copy at [offset] and
  /// marks that range dirty. Unlike `CopyHostBuffer`, this works for every
  /// storage mode, since the shadow copy exists for all of them. Used by
  /// blits, which run on the reactor.
  [[nodiscard]] bool WriteBufferData(const uint8_t* source,
                                     Range source_range,
                                     size_t offset);

  enum class BindingType {
    kArrayBuffer,
    kElementArrayBuffer,
    kUniformBuffer,
    kShaderStorageBuffer,
    kCopyReadBuffer,
    kDispatchIndirectBuffer,
    kDrawIndirectBuffer,
  };

  [[nodiscard]] bool BindAndUploadDataIfNecessary(BindingType type) const;

  /// Records that the GPU may have written this buffer, because a compute
  /// shader bound it as a storage buffer. Until the next
  /// `SyncShadowCopyIfNecessary`, the GL buffer is authoritative and the
  /// shadow copy is stale outside of host writes made since.
  ///
  /// Host writes keep working while the buffer is GPU-written: they land in
  /// the shadow copy and only their dirty range is uploaded.
  void MarkWrittenByGPU() const;

  /// Whether `MarkWrittenByGPU` was called since the shadow copy was last
  /// synchronized.
  bool IsWrittenByGPU() const;

  /// If the GPU wrote this buffer, uploads pending host writes and then reads
  /// the whole buffer back into the shadow copy. Blocks until the GPU
  /// finishes writing it. Must be called on the reactor, after a
  /// `glMemoryBarrier` that covers buffer updates.
  [[nodiscard]] bool SyncShadowCopyIfNecessary() const;

  void Flush(std::optional<Range> range = std::nullopt) const override;

  std::optional<GLuint> GetHandle() const;

  /// Returns the GL buffer used when binding as a uniform buffer.
  std::optional<GLuint> GetUniformHandle() const;

 private:
  std::shared_ptr<ReactorGLES> reactor_;
  std::optional<std::string> label_;
  // Mutable for lazy evaluation.
  mutable std::optional<HandleGLES> handle_;
  // Separate buffer used for uniform data on drivers that require it. See
  // |CapabilitiesGLES::NeedsSeparateUniformBuffers|.
  bool separate_uniform_buffer_ = false;
  mutable std::optional<HandleGLES> uniform_handle_;
  mutable std::unique_ptr<Allocation> backing_store_;
  // Guards dirty_range_. Flush() merges ranges from writer threads (e.g. the
  // UI thread for flutter_gpu host-visible buffers) while
  // BindAndUploadDataIfNecessary() consumes the range on a reactor thread.
  mutable Mutex dirty_range_mutex_;
  mutable std::optional<Range> dirty_range_
      IPLR_GUARDED_BY(dirty_range_mutex_) = std::nullopt;
  mutable std::optional<Range> uniform_dirty_range_
      IPLR_GUARDED_BY(dirty_range_mutex_) = std::nullopt;
  mutable std::atomic<bool> written_by_gpu_ = false;
  mutable bool initialized_ = false;
  mutable bool uniform_initialized_ = false;

  bool UsesUniformBuffer(BindingType type) const;

  // |DeviceBuffer|
  uint8_t* OnGetContents() const override;

  // |DeviceBuffer|
  bool OnCopyHostBuffer(const uint8_t* source,
                        Range source_range,
                        size_t offset) override;

  // |DeviceBuffer|
  bool SetLabel(std::string_view label) override;

  // |DeviceBuffer|
  bool SetLabel(std::string_view label, Range range) override;

  DeviceBufferGLES(const DeviceBufferGLES&) = delete;

  DeviceBufferGLES& operator=(const DeviceBufferGLES&) = delete;
};

}  // namespace impeller

#endif  // FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_DEVICE_BUFFER_GLES_H_
