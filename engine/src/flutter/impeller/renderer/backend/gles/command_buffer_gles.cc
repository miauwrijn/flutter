// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "impeller/renderer/backend/gles/command_buffer_gles.h"

#include <memory>
#include <utility>

#include "impeller/base/config.h"
#include "impeller/renderer/backend/gles/blit_pass_gles.h"
#include "impeller/renderer/backend/gles/compute_pass_gles.h"
#include "impeller/renderer/backend/gles/context_gles.h"
#include "impeller/renderer/backend/gles/render_pass_gles.h"

namespace impeller {
namespace {

/// Invokes a deferred command buffer completion callback exactly once.
///
/// The callback is completed with `kCompleted` when the deferred reactor
/// operation runs. If the operation is discarded before it runs, such as during
/// reactor teardown, the callback is completed with `kError`.
class DeferredCompletionCallback {
 public:
  explicit DeferredCompletionCallback(
      CommandBuffer::CompletionCallback callback)
      : callback_(std::move(callback)) {}

  /// Completes with `kError` unless the callback was already invoked.
  ~DeferredCompletionCallback() { Invoke(CommandBuffer::Status::kError); }

  DeferredCompletionCallback(const DeferredCompletionCallback&) = delete;
  DeferredCompletionCallback& operator=(const DeferredCompletionCallback&) =
      delete;
  DeferredCompletionCallback(DeferredCompletionCallback&&) = delete;
  DeferredCompletionCallback& operator=(DeferredCompletionCallback&&) = delete;

  /// Completes with `kCompleted` when the deferred reactor operation runs.
  void Complete() { Invoke(CommandBuffer::Status::kCompleted); }

 private:
  void Invoke(CommandBuffer::Status status) {
    if (!callback_) {
      return;
    }
    auto callback = std::exchange(callback_, nullptr);
    callback(status);
  }

  CommandBuffer::CompletionCallback callback_;
};

/// Blocks until the GPU finishes the commands the reactor issued so far, so
/// that a completion callback reports GPU completion rather than the reactor
/// consuming the commands. Returns false if the wait fails. Contexts without
/// fences (OpenGL ES 2.0) fall back to glFinish.
bool WaitForGPU(const ProcTableGLES& gl) {
  if (!gl.FenceSync.IsAvailable() || !gl.ClientWaitSync.IsAvailable() ||
      !gl.DeleteSync.IsAvailable()) {
    gl.Finish();
    return true;
  }
  GLsync fence = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (!fence) {
    gl.Finish();
    return true;
  }
  // Flush on the first wait so the fence is guaranteed to signal, then keep
  // waiting in bounded steps.
  constexpr GLuint64 kWaitStepNanoseconds = 1'000'000'000;
  GLbitfield flags = GL_SYNC_FLUSH_COMMANDS_BIT;
  bool signaled = false;
  while (true) {
    const GLenum result = gl.ClientWaitSync(fence, flags, kWaitStepNanoseconds);
    if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED) {
      signaled = true;
      break;
    }
    if (result != GL_TIMEOUT_EXPIRED) {
      break;
    }
    flags = 0;
  }
  gl.DeleteSync(fence);
  return signaled;
}

}  // namespace

CommandBufferGLES::CommandBufferGLES(std::weak_ptr<const Context> context,
                                     std::shared_ptr<ReactorGLES> reactor)
    : CommandBuffer(std::move(context)),
      reactor_(std::move(reactor)),
      is_valid_(reactor_ && reactor_->IsValid()) {}

CommandBufferGLES::~CommandBufferGLES() = default;

// |CommandBuffer|
void CommandBufferGLES::SetLabel(std::string_view label) const {
  // Cannot support.
}

// |CommandBuffer|
bool CommandBufferGLES::IsValid() const {
  return is_valid_;
}

// |CommandBuffer|
bool CommandBufferGLES::OnSubmitCommands(CompletionCallback callback) {
  // The reactor consumes commands on the GL thread and GL synchronizes
  // buffer reuse implicitly, so submissions are tracked at reactor
  // consumption granularity rather than GPU completion. A completion
  // callback, though, runs only once the GPU finished, through a fence.
  std::shared_ptr<GpuSubmissionTracker> tracker;
  uint64_t submission_id = 0;
  if (auto context = context_.lock()) {
    tracker = ContextGLES::Cast(*context).GetMutableSubmissionTracker();
    submission_id = tracker->RecordSubmission();
  }

  if (reactor_->CanReactOnCurrentThread()) {
    auto result = reactor_->React();
    if (tracker) {
      tracker->RecordCompletion(submission_id);
    }
    if (callback) {
      // The callback promises GPU completion, which readbacks rely on.
      result = result && WaitForGPU(reactor_->GetProcTable());
      callback(result ? CommandBuffer::Status::kCompleted
                      : CommandBuffer::Status::kError);
    }
    return result;
  }

  // Submission is accepted even when no GL context is current yet. The
  // reactor keeps previously encoded operations queued on this thread.
  std::shared_ptr<DeferredCompletionCallback> deferred_callback;
  if (callback) {
    deferred_callback =
        std::make_shared<DeferredCompletionCallback>(std::move(callback));
  }
  if (!reactor_->AddOperation(
          [deferred_callback, tracker,
           submission_id](const ReactorGLES& reactor) {
            if (tracker) {
              tracker->RecordCompletion(submission_id);
            }
            if (deferred_callback) {
              // Destroying the callback without completing it reports an
              // error.
              if (WaitForGPU(reactor.GetProcTable())) {
                deferred_callback->Complete();
              }
            }
          },
          /*defer=*/true)) {
    if (tracker) {
      tracker->RecordCompletion(submission_id);
    }
    return false;
  }
  return true;
}

// |CommandBuffer|
void CommandBufferGLES::OnWaitUntilCompleted() {
  reactor_->GetProcTable().Finish();
}

// |CommandBuffer|
void CommandBufferGLES::OnWaitUntilScheduled() {
  reactor_->GetProcTable().Flush();
}

// |CommandBuffer|
std::shared_ptr<RenderPass> CommandBufferGLES::OnCreateRenderPass(
    RenderTarget target) {
  if (!IsValid()) {
    return nullptr;
  }
  auto context = context_.lock();
  if (!context) {
    return nullptr;
  }
  auto pass = std::shared_ptr<RenderPassGLES>(
      new RenderPassGLES(context, target, reactor_));
  if (!pass->IsValid()) {
    return nullptr;
  }
  return pass;
}

// |CommandBuffer|
std::shared_ptr<BlitPass> CommandBufferGLES::OnCreateBlitPass() {
  if (!IsValid()) {
    return nullptr;
  }
  auto pass = std::shared_ptr<BlitPassGLES>(new BlitPassGLES(reactor_));
  if (!pass->IsValid()) {
    return nullptr;
  }
  return pass;
}

// |CommandBuffer|
std::shared_ptr<ComputePass> CommandBufferGLES::OnCreateComputePass() {
  // Compute is core in OpenGL ES 3.1 and OpenGL 4.3.
  if (!IsValid()) {
    return nullptr;
  }
  auto context = context_.lock();
  if (!context || !context->GetCapabilities()->SupportsCompute()) {
    return nullptr;
  }
  auto pass =
      std::shared_ptr<ComputePassGLES>(new ComputePassGLES(context, reactor_));
  if (!pass->IsValid()) {
    return nullptr;
  }
  return pass;
}

}  // namespace impeller
