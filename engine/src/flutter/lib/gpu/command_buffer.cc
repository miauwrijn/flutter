// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "flutter/lib/gpu/command_buffer.h"

#include <algorithm>
#include <cstring>

#include "dart_api.h"
#include "fml/closure.h"
#include "fml/logging.h"
#include "fml/make_copyable.h"
#include "impeller/core/buffer_view.h"
#include "impeller/renderer/command_buffer.h"
#include "impeller/renderer/render_pass.h"
#include "lib/ui/ui_dart_state.h"
#include "tonic/converter/dart_converter.h"

namespace flutter {
namespace gpu {

IMPLEMENT_WRAPPERTYPEINFO(flutter_gpu, CommandBuffer);

CommandBuffer::CommandBuffer(
    std::shared_ptr<impeller::Context> context,
    std::shared_ptr<impeller::CommandBuffer> command_buffer)
    : context_(std::move(context)),
      command_buffer_(std::move(command_buffer)) {}

CommandBuffer::~CommandBuffer() {
  // A command buffer that is discarded without being submitted never runs.
  // Report that to everything waiting on it, such as pending readbacks, so
  // that nothing waits forever.
  if (!submitted_) {
    for (auto& callback : completion_callbacks_) {
      callback(impeller::CommandBuffer::Status::kError);
    }
  }
}

bool CommandBuffer::Encodable::EncodeCommands() const {
  if (render_pass) {
    return render_pass->EncodeCommands();
  }
  if (blit_pass) {
    return blit_pass->EncodeCommands();
  }
  if (compute_pass) {
    return compute_pass->EncodeCommands();
  }
  return false;
}

std::shared_ptr<impeller::CommandBuffer> CommandBuffer::GetCommandBuffer() {
  return command_buffer_;
}

std::shared_ptr<impeller::RenderPass> CommandBuffer::CreateRenderPass(
    const impeller::RenderTarget& render_target) {
  if (!EndOpenPass()) {
    return nullptr;
  }
  auto render_pass = command_buffer_->CreateRenderPass(render_target);
  if (!render_pass) {
    return nullptr;
  }
  Encodable encodable;
  encodable.render_pass = render_pass;
  encodables_.push_back(std::move(encodable));
  has_open_pass_ = true;
  return render_pass;
}

bool CommandBuffer::EndRenderPass(const impeller::RenderPass* render_pass) {
  if (!render_pass || !has_open_pass_ ||
      encodables_.back().render_pass.get() != render_pass) {
    return true;
  }
  return EndOpenPass();
}

std::shared_ptr<impeller::ComputePass> CommandBuffer::CreateComputePass() {
  if (!EndOpenPass()) {
    return nullptr;
  }
  auto compute_pass = command_buffer_->CreateComputePass();
  if (!compute_pass) {
    return nullptr;
  }
  Encodable encodable;
  encodable.compute_pass = compute_pass;
  encodables_.push_back(std::move(encodable));
  has_open_pass_ = true;
  return compute_pass;
}

absl::Status CommandBuffer::EndComputePass(
    const impeller::ComputePass* compute_pass) {
  FML_DCHECK(compute_pass);
  if (!has_open_pass_ ||
      encodables_.back().compute_pass.get() != compute_pass) {
    return absl::OkStatus();
  }
  if (!EndOpenPass()) {
    return absl::InternalError("Failed to encode the compute pass.");
  }
  return absl::OkStatus();
}

const std::shared_ptr<impeller::Context>& CommandBuffer::GetContext() const {
  return context_;
}

bool CommandBuffer::EndOpenPass() {
  if (!has_open_pass_) {
    return true;
  }
  has_open_pass_ = false;
  // GLES passes are encoded at submit, on the raster thread.
  if (context_->GetBackendType() == impeller::Context::BackendType::kOpenGLES) {
    return true;
  }
  if (!encodables_.back().EncodeCommands()) {
    encoding_failed_ = true;
    return false;
  }
  return true;
}

std::shared_ptr<impeller::BlitPass> CommandBuffer::GetOrCreateBlitPass() {
  // Consecutive copies share one blit pass.
  if (has_open_pass_ && encodables_.back().blit_pass) {
    return encodables_.back().blit_pass;
  }
  if (!EndOpenPass()) {
    return nullptr;
  }
  auto blit_pass = command_buffer_->CreateBlitPass();
  if (!blit_pass) {
    return nullptr;
  }
  Encodable encodable;
  encodable.blit_pass = blit_pass;
  encodables_.push_back(std::move(encodable));
  has_open_pass_ = true;
  return blit_pass;
}

bool CommandBuffer::CopyBufferToTexture(DeviceBuffer& source,
                                        size_t source_offset,
                                        size_t source_length,
                                        Texture& destination,
                                        impeller::IRect destination_region,
                                        uint32_t mip_level,
                                        uint32_t slice) {
  auto blit_pass = GetOrCreateBlitPass();
  if (!blit_pass) {
    return false;
  }
  impeller::BufferView source_view(
      source.GetBuffer(), impeller::Range(source_offset, source_length));
  return blit_pass->AddCopy(
      std::move(source_view), destination.GetTexture(), destination_region,
      /*label=*/"CommandBuffer.copyBufferToTexture", mip_level, slice);
}

bool CommandBuffer::CopyTextureToBuffer(Texture& source,
                                        impeller::IRect source_region,
                                        DeviceBuffer& destination,
                                        size_t destination_offset) {
  auto blit_pass = GetOrCreateBlitPass();
  if (!blit_pass) {
    return false;
  }
  return blit_pass->AddCopy(source.GetTexture(), destination.GetBuffer(),
                            source_region, destination_offset,
                            "CommandBuffer.copyTextureToBuffer");
}

void CommandBuffer::SetReadbackBufferPool(
    std::shared_ptr<ReadbackBufferPool> pool) {
  readback_buffer_pool_ = std::move(pool);
}

const std::shared_ptr<ReadbackBufferPool>&
CommandBuffer::GetReadbackBufferPool() const {
  return readback_buffer_pool_;
}

std::shared_ptr<impeller::DeviceBuffer> CommandBuffer::CreateReadbackBuffer(
    size_t length) {
  if (readback_buffer_pool_) {
    return readback_buffer_pool_->Acquire(*context_->GetResourceAllocator(),
                                          length);
  }
  impeller::DeviceBufferDescriptor desc;
  desc.storage_mode = impeller::StorageMode::kHostVisible;
  // Zero-sized allocations are not portable, and an empty read still
  // completes in order with the command buffer.
  desc.size = std::max<size_t>(length, 1u);
  desc.readback = true;
  auto buffer = context_->GetResourceAllocator()->CreateBuffer(desc);
  if (buffer) {
    buffer->SetLabel("Flutter GPU readback");
  }
  return buffer;
}

std::shared_ptr<impeller::DeviceBuffer> CommandBuffer::RecordBufferReadback(
    DeviceBuffer& source,
    size_t offset,
    size_t length) {
  auto blit_pass = GetOrCreateBlitPass();
  if (!blit_pass) {
    return nullptr;
  }
  auto readback = CreateReadbackBuffer(length);
  if (!readback) {
    return nullptr;
  }
  if (!blit_pass->AddCopy(impeller::BufferView(source.GetBuffer(),
                                               impeller::Range(offset, length)),
                          readback, /*destination_offset=*/0,
                          "CommandBuffer.readBuffer")) {
    return nullptr;
  }
  return readback;
}

std::shared_ptr<impeller::DeviceBuffer> CommandBuffer::RecordTextureReadback(
    Texture& source,
    impeller::IRect source_region,
    size_t length) {
  auto blit_pass = GetOrCreateBlitPass();
  if (!blit_pass) {
    return nullptr;
  }
  auto readback = CreateReadbackBuffer(length);
  if (!readback) {
    return nullptr;
  }
  if (!blit_pass->AddCopy(source.GetTexture(), readback, source_region,
                          /*destination_offset=*/0,
                          "CommandBuffer.readTexture")) {
    return nullptr;
  }
  return readback;
}

bool CommandBuffer::CopyTextureToTexture(Texture& source,
                                         Texture& destination,
                                         impeller::IRect source_region,
                                         impeller::IPoint destination_origin) {
  auto blit_pass = GetOrCreateBlitPass();
  if (!blit_pass) {
    return false;
  }
  return blit_pass->AddCopy(source.GetTexture(), destination.GetTexture(),
                            source_region, destination_origin,
                            "CommandBuffer.copyTextureToTexture");
}

bool CommandBuffer::AddCompletionCallback(
    impeller::CommandBuffer::CompletionCallback completion_callback) {
  if (submitted_) {
    return false;
  }
  if (completion_callback) {
    completion_callbacks_.push_back(std::move(completion_callback));
  }
  return true;
}

bool CommandBuffer::Submit() {
  return CommandBuffer::Submit({});
}

bool CommandBuffer::Submit(
    const impeller::CommandBuffer::CompletionCallback& completion_callback) {
  if (submitted_) {
    return false;
  }
  submitted_ = true;

  // On Metal and Vulkan this encodes the last pass. Every earlier pass was
  // encoded when it ended.
  EndOpenPass();

  std::vector<impeller::CommandBuffer::CompletionCallback> callbacks =
      std::move(completion_callbacks_);
  if (completion_callback) {
    callbacks.push_back(completion_callback);
  }
  impeller::CommandBuffer::CompletionCallback combined_completion_callback;
  if (!callbacks.empty()) {
    combined_completion_callback =
        [callbacks = std::move(callbacks)](
            impeller::CommandBuffer::Status status) mutable {
          for (auto& callback : callbacks) {
            callback(status);
          }
        };
  }

  // For the GLES backend, command queue submission just flushes the reactor,
  // which needs to happen on the raster thread.
  if (context_->GetBackendType() == impeller::Context::BackendType::kOpenGLES) {
    auto dart_state = flutter::UIDartState::Current();
    auto& task_runners = dart_state->GetTaskRunners();

    task_runners.GetRasterTaskRunner()->PostTask(
        fml::MakeCopyable([context = context_, command_buffer = command_buffer_,
                           completion_callback = combined_completion_callback,
                           encodables = encodables_]() mutable {
          for (auto& encodable : encodables) {
            if (!encodable.EncodeCommands()) {
              if (completion_callback) {
                completion_callback(impeller::CommandBuffer::Status::kError);
              }
              context->DisposeThreadLocalCachedResources();
              return;
            }
          }

          auto status = context->GetCommandQueue()->Submit({command_buffer},
                                                           completion_callback);
          if (!status.ok() && completion_callback) {
            completion_callback(impeller::CommandBuffer::Status::kError);
          }
          context->DisposeThreadLocalCachedResources();
        }));
    return true;
  }

  if (encoding_failed_) {
    if (combined_completion_callback) {
      combined_completion_callback(impeller::CommandBuffer::Status::kError);
    }
    return false;
  }

  auto status = context_->GetCommandQueue()->Submit(
      {command_buffer_}, combined_completion_callback);
  context_->DisposeThreadLocalCachedResources();
  if (!status.ok() && combined_completion_callback) {
    combined_completion_callback(impeller::CommandBuffer::Status::kError);
  }
  return status.ok();
}

}  // namespace gpu
}  // namespace flutter

//----------------------------------------------------------------------------
/// Exports
///

bool InternalFlutterGpu_CommandBuffer_Initialize(
    Dart_Handle wrapper,
    flutter::gpu::Context* contextWrapper) {
  auto res = fml::MakeRefCounted<flutter::gpu::CommandBuffer>(
      contextWrapper->GetContextShared(),
      contextWrapper->GetContext().CreateCommandBuffer());
  res->SetReadbackBufferPool(contextWrapper->GetReadbackBufferPool());
  res->AssociateWithDartWrapper(wrapper);

  return true;
}

Dart_Handle InternalFlutterGpu_CommandBuffer_Submit(
    flutter::gpu::CommandBuffer* wrapper,
    Dart_Handle completion_callback) {
  if (Dart_IsNull(completion_callback)) {
    bool success = wrapper->Submit();
    if (!success) {
      return tonic::ToDart("Failed to submit CommandBuffer");
    }
    return Dart_Null();
  }

  if (!Dart_IsClosure(completion_callback)) {
    return tonic::ToDart("Completion callback must be a function");
  }

  auto dart_state = flutter::UIDartState::Current();
  auto& task_runners = dart_state->GetTaskRunners();

  auto persistent_completion_callback =
      std::make_unique<tonic::DartPersistentValue>(dart_state,
                                                   completion_callback);

  auto ui_task_completion_callback = fml::MakeCopyable(
      [callback = std::move(persistent_completion_callback),
       task_runners](impeller::CommandBuffer::Status status) mutable {
        bool success = status != impeller::CommandBuffer::Status::kError;

        auto ui_completion_task = fml::MakeCopyable(
            [callback = std::move(callback), success]() mutable {
              auto dart_state = callback->dart_state().lock();
              if (!dart_state) {
                // The root isolate could have died in the meantime.
                return;
              }
              tonic::DartState::Scope scope(dart_state);

              tonic::DartInvoke(callback->Get(), {tonic::ToDart(success)});

              // callback is associated with the Dart isolate and must be
              // deleted on the UI thread.
              callback.reset();
            });
        task_runners.GetUITaskRunner()->PostTask(ui_completion_task);
      });
  bool success = wrapper->Submit(ui_task_completion_callback);
  if (!success) {
    return tonic::ToDart("Failed to submit CommandBuffer");
  }
  return Dart_Null();
}

namespace {

/// Completes a readback on the UI thread: calls [callback] with a Uint8List
/// holding the first [length] bytes of [readback], or with null if the command
/// buffer failed or was discarded.
impeller::CommandBuffer::CompletionCallback MakeReadbackCompletion(
    Dart_Handle completion_callback,
    std::shared_ptr<impeller::DeviceBuffer> readback,
    size_t length,
    std::shared_ptr<flutter::gpu::ReadbackBufferPool> pool) {
  auto dart_state = flutter::UIDartState::Current();
  auto ui_task_runner = dart_state->GetTaskRunners().GetUITaskRunner();
  auto persistent_callback = std::make_unique<tonic::DartPersistentValue>(
      dart_state, completion_callback);

  return fml::MakeCopyable([callback = std::move(persistent_callback),
                            readback = std::move(readback), length,
                            ui_task_runner, pool = std::move(pool)](
                               impeller::CommandBuffer::Status status) mutable {
    // Called at most once, from the thread that observes completion.
    if (!callback) {
      return;
    }
    const bool success = status == impeller::CommandBuffer::Status::kCompleted;
    ui_task_runner->PostTask(fml::MakeCopyable(
        [callback = std::move(callback), readback = std::move(readback), length,
         success, pool = std::move(pool)]() mutable {
          // The command buffer has finished with the buffer, so once its bytes
          // are copied out it can serve the next readback.
          fml::ScopedCleanupClosure release_readback([&readback, &pool]() {
            if (pool) {
              pool->Release(std::move(readback));
            }
          });
          auto dart_state = callback->dart_state().lock();
          if (!dart_state) {
            // The isolate could have died in the meantime.
            return;
          }
          tonic::DartState::Scope scope(dart_state);

          Dart_Handle result = Dart_Null();
          const uint8_t* contents = nullptr;
          if (success) {
            // Make GPU writes to non-coherent memory visible to the host.
            readback->Invalidate(impeller::Range(0, length));
            contents = readback->OnGetContents();
          }
          if (contents) {
            result = Dart_NewTypedData(Dart_TypedData_kUint8, length);
            if (!Dart_IsError(result) && length > 0) {
              Dart_TypedData_Type type;
              void* data = nullptr;
              intptr_t data_length = 0;
              Dart_Handle acquired =
                  Dart_TypedDataAcquireData(result, &type, &data, &data_length);
              if (Dart_IsError(acquired)) {
                result = Dart_Null();
              } else {
                std::memcpy(data, contents, length);
                Dart_TypedDataReleaseData(result);
              }
            }
            if (Dart_IsError(result)) {
              result = Dart_Null();
            }
          }

          tonic::DartInvoke(callback->Get(), {result});

          // The callback belongs to the isolate and must be released on
          // the UI thread.
          callback.reset();
        }));
  });
}

}  // namespace

Dart_Handle InternalFlutterGpu_CommandBuffer_ReadBuffer(
    flutter::gpu::CommandBuffer* wrapper,
    flutter::gpu::DeviceBuffer* source,
    int source_offset_in_bytes,
    int source_length_in_bytes,
    Dart_Handle completion_callback) {
  if (!Dart_IsClosure(completion_callback)) {
    return tonic::ToDart("Completion callback must be a function");
  }
  if (source_offset_in_bytes < 0 || source_length_in_bytes < 0) {
    return tonic::ToDart("Readback range must be non-negative");
  }
  auto readback = wrapper->RecordBufferReadback(
      *source, static_cast<size_t>(source_offset_in_bytes),
      static_cast<size_t>(source_length_in_bytes));
  if (!readback) {
    return tonic::ToDart("Failed to record the buffer readback");
  }
  if (!wrapper->AddCompletionCallback(
          MakeReadbackCompletion(completion_callback, std::move(readback),
                                 static_cast<size_t>(source_length_in_bytes),
                                 wrapper->GetReadbackBufferPool()))) {
    return tonic::ToDart("The CommandBuffer was already submitted");
  }
  return Dart_Null();
}

Dart_Handle InternalFlutterGpu_CommandBuffer_ReadTexture(
    flutter::gpu::CommandBuffer* wrapper,
    flutter::gpu::Texture* source,
    int source_x,
    int source_y,
    int source_width,
    int source_height,
    int length_in_bytes,
    Dart_Handle completion_callback) {
  if (!Dart_IsClosure(completion_callback)) {
    return tonic::ToDart("Completion callback must be a function");
  }
  if (length_in_bytes < 0) {
    return tonic::ToDart("Readback length must be non-negative");
  }
  auto readback = wrapper->RecordTextureReadback(
      *source,
      impeller::IRect::MakeXYWH(source_x, source_y, source_width,
                                source_height),
      static_cast<size_t>(length_in_bytes));
  if (!readback) {
    return tonic::ToDart("Failed to record the texture readback");
  }
  if (!wrapper->AddCompletionCallback(
          MakeReadbackCompletion(completion_callback, std::move(readback),
                                 static_cast<size_t>(length_in_bytes),
                                 wrapper->GetReadbackBufferPool()))) {
    return tonic::ToDart("The CommandBuffer was already submitted");
  }
  return Dart_Null();
}

Dart_Handle InternalFlutterGpu_CommandBuffer_CopyBufferToTexture(
    flutter::gpu::CommandBuffer* command_buffer,
    flutter::gpu::DeviceBuffer* source,
    int source_offset_in_bytes,
    int source_length_in_bytes,
    flutter::gpu::Texture* destination,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    int mip_level,
    int slice) {
  if (!command_buffer->CopyBufferToTexture(
          *source, static_cast<size_t>(source_offset_in_bytes),
          static_cast<size_t>(source_length_in_bytes), *destination,
          impeller::IRect::MakeXYWH(destination_x, destination_y,
                                    destination_width, destination_height),
          static_cast<uint32_t>(mip_level), static_cast<uint32_t>(slice))) {
    return tonic::ToDart("Failed to append copyBufferToTexture");
  }
  return Dart_Null();
}

Dart_Handle InternalFlutterGpu_CommandBuffer_CopyTextureToBuffer(
    flutter::gpu::CommandBuffer* command_buffer,
    flutter::gpu::Texture* source,
    int source_x,
    int source_y,
    int source_width,
    int source_height,
    flutter::gpu::DeviceBuffer* destination,
    int destination_offset_in_bytes) {
  if (!command_buffer->CopyTextureToBuffer(
          *source,
          impeller::IRect::MakeXYWH(source_x, source_y, source_width,
                                    source_height),
          *destination, static_cast<size_t>(destination_offset_in_bytes))) {
    return tonic::ToDart("Failed to append copyTextureToBuffer");
  }
  return Dart_Null();
}

Dart_Handle InternalFlutterGpu_CommandBuffer_CopyTextureToTexture(
    flutter::gpu::CommandBuffer* command_buffer,
    flutter::gpu::Texture* source,
    flutter::gpu::Texture* destination,
    int source_x,
    int source_y,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y) {
  if (!command_buffer->CopyTextureToTexture(
          *source, *destination,
          impeller::IRect::MakeXYWH(source_x, source_y, source_width,
                                    source_height),
          impeller::IPoint(destination_x, destination_y))) {
    return tonic::ToDart("Failed to append copyTextureToTexture");
  }
  return Dart_Null();
}
