// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "impeller/renderer/backend/gles/device_buffer_gles.h"

#include <cstring>
#include <memory>

#include "impeller/base/allocation.h"
#include "impeller/base/config.h"
#include "impeller/base/validation.h"

namespace impeller {

DeviceBufferGLES::DeviceBufferGLES(DeviceBufferDescriptor desc,
                                   std::shared_ptr<ReactorGLES> reactor,
                                   std::unique_ptr<Allocation> backing_store)
    : DeviceBuffer(desc),
      reactor_(std::move(reactor)),
      backing_store_(std::move(backing_store)) {
  if (reactor_) {
    const auto& capabilities = reactor_->GetProcTable().GetCapabilities();
    separate_uniform_buffer_ =
        capabilities && capabilities->NeedsSeparateUniformBuffers();
  }
}

// |DeviceBuffer|
DeviceBufferGLES::~DeviceBufferGLES() {
  if (handle_.has_value() && !handle_->IsDead()) {
    reactor_->CollectHandle(*handle_);
  }
  if (uniform_handle_.has_value() && !uniform_handle_->IsDead()) {
    reactor_->CollectHandle(*uniform_handle_);
  }
}

// |DeviceBuffer|
uint8_t* DeviceBufferGLES::OnGetContents() const {
  if (!reactor_) {
    return nullptr;
  }
  return backing_store_->GetBuffer();
}

// |DeviceBuffer|
bool DeviceBufferGLES::OnCopyHostBuffer(const uint8_t* source,
                                        Range source_range,
                                        size_t offset) {
  if (!reactor_) {
    return false;
  }

  if (offset + source_range.length >
      backing_store_->GetLength().GetByteSize()) {
    return false;
  }

  std::memmove(backing_store_->GetBuffer() + offset,
               source + source_range.offset, source_range.length);
  Flush(Range{offset, source_range.length});

  return true;
}

bool DeviceBufferGLES::WriteBufferData(const uint8_t* source,
                                       Range source_range,
                                       size_t offset) {
  if (source_range.length == 0u) {
    return true;
  }
  if (source == nullptr) {
    return false;
  }
  return OnCopyHostBuffer(source, source_range, offset);
}

std::optional<GLuint> DeviceBufferGLES::GetHandle() const {
  if (handle_.has_value()) {
    return reactor_->GetGLHandle(*handle_);
  } else {
    return std::nullopt;
  }
}

std::optional<GLuint> DeviceBufferGLES::GetUniformHandle() const {
  if (!separate_uniform_buffer_) {
    return GetHandle();
  }
  if (uniform_handle_.has_value()) {
    return reactor_->GetGLHandle(*uniform_handle_);
  } else {
    return std::nullopt;
  }
}

bool DeviceBufferGLES::UsesUniformBuffer(BindingType type) const {
  return separate_uniform_buffer_ && type == BindingType::kUniformBuffer;
}

static void MergeRange(std::optional<Range>& dirty_range, Range range) {
  if (dirty_range.has_value()) {
    dirty_range = dirty_range->Merge(range);
  } else {
    dirty_range = range;
  }
}

void DeviceBufferGLES::Flush(std::optional<Range> range) const {
  Lock lock(dirty_range_mutex_);
  if (!range.has_value()) {
    dirty_range_ = Range{
        0, static_cast<size_t>(backing_store_->GetLength().GetByteSize())};
    if (separate_uniform_buffer_) {
      uniform_dirty_range_ = dirty_range_;
    }
  } else {
    MergeRange(dirty_range_, range.value());
    if (separate_uniform_buffer_) {
      MergeRange(uniform_dirty_range_, range.value());
    }
  }
}

static GLenum ToTarget(DeviceBufferGLES::BindingType type) {
  switch (type) {
    case DeviceBufferGLES::BindingType::kArrayBuffer:
      return GL_ARRAY_BUFFER;
    case DeviceBufferGLES::BindingType::kElementArrayBuffer:
      return GL_ELEMENT_ARRAY_BUFFER;
    case DeviceBufferGLES::BindingType::kUniformBuffer:
      return GL_UNIFORM_BUFFER;
    case DeviceBufferGLES::BindingType::kShaderStorageBuffer:
      return GL_SHADER_STORAGE_BUFFER;
    case DeviceBufferGLES::BindingType::kCopyReadBuffer:
      return GL_COPY_READ_BUFFER;
  }
  FML_UNREACHABLE();
}

bool DeviceBufferGLES::BindAndUploadDataIfNecessary(BindingType type) const {
  if (!reactor_) {
    return false;
  }

  // The separate uniform buffer is filled from the shadow copy, which is
  // stale while the GPU-written main buffer is authoritative.
  if (UsesUniformBuffer(type) && !SyncShadowCopyIfNecessary()) {
    return false;
  }

  const bool uses_uniform_buffer = UsesUniformBuffer(type);
  std::optional<HandleGLES>& handle =
      uses_uniform_buffer ? uniform_handle_ : handle_;
  bool& initialized = uses_uniform_buffer ? uniform_initialized_ : initialized_;

  if (!handle.has_value()) {
    handle = reactor_->CreateUntrackedHandle(HandleType::kBuffer);
#ifdef IMPELLER_DEBUG
    if (handle.has_value() && label_.has_value()) {
      reactor_->SetDebugLabel(*handle, *label_);
    }
#endif
  }

  auto buffer = reactor_->GetGLHandle(*handle);
  if (!buffer.has_value()) {
    return false;
  }

  const auto target_type = ToTarget(type);
  const auto& gl = reactor_->GetProcTable();

  gl.BindBuffer(target_type, buffer.value());
  if (!initialized) {
    gl.BufferData(target_type, backing_store_->GetLength().GetByteSize(),
                  nullptr, GL_DYNAMIC_DRAW);
    initialized = true;
  }

  // Take and clear the dirty range BEFORE uploading. A Flush() from another
  // thread during the upload then merges into a fresh dirty range that the
  // next bind uploads, instead of being silently discarded by a clear that
  // runs after the upload.
  std::optional<Range> dirty;
  {
    Lock lock(dirty_range_mutex_);
    std::swap(uses_uniform_buffer ? uniform_dirty_range_ : dirty_range_, dirty);
  }
  if (dirty.has_value()) {
    gl.BufferSubData(target_type, dirty->offset, dirty->length,
                     backing_store_->GetBuffer() + dirty->offset);
  }

  return true;
}

// |DeviceBuffer|
bool DeviceBufferGLES::SetLabel(std::string_view label) {
#ifdef IMPELLER_DEBUG
  label_ = label;
  if (handle_.has_value()) {
    reactor_->SetDebugLabel(*handle_, label);
  }
  if (uniform_handle_.has_value()) {
    reactor_->SetDebugLabel(*uniform_handle_, label);
  }
#endif  // IMPELLER_DEBUG
  return true;
}

// |DeviceBuffer|
bool DeviceBufferGLES::SetLabel(std::string_view label, Range range) {
  // Cannot support debug label on the range. Set the label for the entire
  // range.
  return SetLabel(label);
}

const uint8_t* DeviceBufferGLES::GetBufferData() const {
  if (!SyncShadowCopyIfNecessary()) {
    VALIDATION_LOG << "Could not read a GPU-written buffer back into its "
                      "shadow copy. Its contents are stale.";
  }
  return backing_store_->GetBuffer();
}

void DeviceBufferGLES::MarkWrittenByGPU() const {
  written_by_gpu_.store(true);
}

bool DeviceBufferGLES::IsWrittenByGPU() const {
  return written_by_gpu_.load();
}

bool DeviceBufferGLES::SyncShadowCopyIfNecessary() const {
  if (!written_by_gpu_.load()) {
    return true;
  }
  if (!reactor_ || !handle_.has_value()) {
    // Nothing was ever uploaded, so the GPU cannot have written it.
    written_by_gpu_.store(false);
    return true;
  }
  const ProcTableGLES& gl = reactor_->GetProcTable();
  if (!gl.MapBufferRange.IsAvailable() || !gl.UnmapBuffer.IsAvailable()) {
    return false;
  }
  // Upload host writes made while the buffer was GPU-written, so the read
  // below returns them along with the GPU's writes.
  if (!BindAndUploadDataIfNecessary(BindingType::kCopyReadBuffer)) {
    return false;
  }
  const size_t size = backing_store_->GetLength().GetByteSize();
  if (size > 0) {
    void* mapped =
        gl.MapBufferRange(GL_COPY_READ_BUFFER, 0, size, GL_MAP_READ_BIT);
    if (!mapped) {
      return false;
    }
    std::memcpy(backing_store_->GetBuffer(), mapped, size);
    gl.UnmapBuffer(GL_COPY_READ_BUFFER);
  }
  gl.BindBuffer(GL_COPY_READ_BUFFER, 0);
  written_by_gpu_.store(false);
  // The separate uniform buffer has to pick up what the GPU wrote.
  if (separate_uniform_buffer_) {
    Lock lock(dirty_range_mutex_);
    uniform_dirty_range_ = Range{0, size};
  }
  return true;
}

void DeviceBufferGLES::UpdateBufferData(
    const std::function<void(uint8_t* data, size_t length)>&
        update_buffer_data) {
  // The update writes part of the shadow copy, and then the whole of it is
  // uploaded, so the rest has to be current first.
  if (!SyncShadowCopyIfNecessary()) {
    VALIDATION_LOG << "Could not read a GPU-written buffer back into its "
                      "shadow copy before updating it.";
  }
  if (update_buffer_data) {
    update_buffer_data(backing_store_->GetBuffer(),
                       backing_store_->GetLength().GetByteSize());
    Flush(Range{
        0, static_cast<size_t>(backing_store_->GetLength().GetByteSize())});
  }
}

}  // namespace impeller
