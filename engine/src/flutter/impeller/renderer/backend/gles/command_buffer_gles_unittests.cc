// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <atomic>
#include <memory>
#include <vector>

#include "flutter/fml/mapping.h"
#include "flutter/testing/testing.h"  // IWYU pragma: keep
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "impeller/core/formats.h"
#include "impeller/renderer/backend/gles/context_gles.h"
#include "impeller/renderer/backend/gles/reactor_gles.h"
#include "impeller/renderer/backend/gles/test/mock_gles.h"
#include "impeller/renderer/backend/gles/texture_gles.h"
#include "impeller/renderer/blit_pass.h"
#include "impeller/renderer/command_queue.h"
#include "impeller/renderer/context.h"

namespace impeller {
namespace testing {
namespace {

using ::testing::_;
using ::testing::NiceMock;

class ToggleWorker final : public ReactorGLES::Worker {
 public:
  explicit ToggleWorker(bool allowed) : allowed_(allowed) {}

  bool CanReactorReactOnCurrentThreadNow(
      const ReactorGLES& reactor) const override {
    return allowed_.load();
  }

  void SetAllowed(bool allowed) { allowed_.store(allowed); }

 private:
  std::atomic<bool> allowed_;
};

std::shared_ptr<ContextGLES> CreateFakeGLESContext() {
  auto gl = std::make_unique<ProcTableGLES>(kMockResolverGLES);
  return ContextGLES::Create(Flags{}, std::move(gl),
                             std::vector<std::shared_ptr<fml::Mapping>>{},
                             /*enable_gpu_tracing=*/false);
}

}  // namespace

TEST(CommandBufferGLES,
     SubmitWithoutCurrentContextIsAcceptedAndCompletesAfterReaction) {
  auto mock_gles = MockGLES::Init();
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(false);
  context->AddReactorWorker(worker);

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);

  bool callback_called = false;
  CommandBuffer::Status callback_status = CommandBuffer::Status::kError;
  auto status = context_base->GetCommandQueue()->Submit(
      {command_buffer}, [&](CommandBuffer::Status status) {
        callback_called = true;
        callback_status = status;
      });

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(callback_called);

  worker->SetAllowed(true);
  EXPECT_TRUE(context->GetReactor()->React());
  EXPECT_TRUE(callback_called);
  EXPECT_EQ(callback_status, CommandBuffer::Status::kCompleted);
}

TEST(CommandBufferGLES, SubmitWithoutCallbackIsAcceptedWithoutCurrentContext) {
  auto mock_gles = MockGLES::Init();
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(false);
  context->AddReactorWorker(worker);

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);

  EXPECT_TRUE(context_base->GetCommandQueue()->Submit({command_buffer}).ok());
}

TEST(CommandBufferGLES, DeferredSubmitCallbackErrorsIfReactorIsDestroyed) {
  bool callback_called = false;
  CommandBuffer::Status callback_status = CommandBuffer::Status::kCompleted;

  {
    auto mock_gles = MockGLES::Init();
    auto context = CreateFakeGLESContext();
    auto context_base = std::static_pointer_cast<Context>(context);
    auto worker = std::make_shared<ToggleWorker>(false);
    context->AddReactorWorker(worker);

    auto command_buffer = context_base->CreateCommandBuffer();
    ASSERT_TRUE(command_buffer);
    auto status = context_base->GetCommandQueue()->Submit(
        {command_buffer}, [&](CommandBuffer::Status status) {
          callback_called = true;
          callback_status = status;
        });

    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(callback_called);
  }

  EXPECT_TRUE(callback_called);
  EXPECT_EQ(callback_status, CommandBuffer::Status::kError);
}

TEST(CommandBufferGLES, DeferredSubmitCompletesAfterPreviouslyQueuedWork) {
  auto mock_gles = MockGLES::Init();
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(false);
  context->AddReactorWorker(worker);

  std::vector<int> order;
  ASSERT_TRUE(context->GetReactor()->AddOperation(
      [&](const ReactorGLES& reactor) { order.push_back(1); },
      /*defer=*/true));

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);
  auto status = context_base->GetCommandQueue()->Submit(
      {command_buffer},
      [&](CommandBuffer::Status status) { order.push_back(2); });

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(order.empty());

  worker->SetAllowed(true);
  EXPECT_TRUE(context->GetReactor()->React());
  EXPECT_THAT(order, ::testing::ElementsAre(1, 2));
}

TEST(CommandBufferGLES, LaterCurrentSubmitDrainsPreviouslyDeferredWork) {
  auto mock_gles = MockGLES::Init();
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(false);
  context->AddReactorWorker(worker);

  std::vector<int> order;
  ASSERT_TRUE(context->GetReactor()->AddOperation(
      [&](const ReactorGLES& reactor) { order.push_back(1); },
      /*defer=*/true));

  auto deferred_command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(deferred_command_buffer);
  EXPECT_TRUE(
      context_base->GetCommandQueue()->Submit({deferred_command_buffer}).ok());
  EXPECT_TRUE(order.empty());

  worker->SetAllowed(true);
  auto current_command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(current_command_buffer);
  EXPECT_TRUE(
      context_base->GetCommandQueue()
          ->Submit({current_command_buffer},
                   [&](CommandBuffer::Status status) { order.push_back(2); })
          .ok());
  EXPECT_THAT(order, ::testing::ElementsAre(1, 2));
}

TEST(CommandBufferGLES, BufferToTextureBlitCanBeSubmittedBeforeContextCurrent) {
  auto mock_gles_impl = std::make_unique<NiceMock<MockGLESImpl>>();
  auto& mock_gles_impl_ref = *mock_gles_impl;
  auto mock_gles = MockGLES::Init(std::move(mock_gles_impl), std::nullopt,
                                  "OpenGL ES 3.2 v1.r18p0-01rel0", "Mali-G72");
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(false);
  context->AddReactorWorker(worker);

  constexpr GLuint kTextureHandle = 1234;
  bool texture_generated = false;
  bool texture_uploaded = false;
  EXPECT_CALL(mock_gles_impl_ref, GenTextures(1, _))
      .WillOnce([&](GLsizei size, GLuint* textures) {
        texture_generated = true;
        textures[0] = kTextureHandle;
      });
  EXPECT_CALL(mock_gles_impl_ref, BindTexture(GL_TEXTURE_2D, 0u)).Times(1);
  EXPECT_CALL(mock_gles_impl_ref, BindTexture(GL_TEXTURE_2D, kTextureHandle))
      .Times(::testing::AtLeast(1));
  EXPECT_CALL(mock_gles_impl_ref,
              TexImage2D(GL_TEXTURE_2D, 0, _, 2, 2, 0, _, _, nullptr))
      .Times(1);
  EXPECT_CALL(mock_gles_impl_ref,
              TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 2, _, _, _))
      .WillOnce([&](GLenum target, GLint level, GLint xoffset, GLint yoffset,
                    GLsizei width, GLsizei height, GLenum format, GLenum type,
                    const void* pixels) { texture_uploaded = true; });

  TextureDescriptor texture_descriptor;
  texture_descriptor.size = {2, 2};
  texture_descriptor.format = PixelFormat::kR8G8B8A8UNormInt;
  auto texture = context_base->GetResourceAllocator()->CreateTexture(
      texture_descriptor, /*threadsafe=*/true);
  ASSERT_TRUE(texture);

  const uint8_t pixels[16] = {};
  auto buffer =
      context_base->GetResourceAllocator()->CreateBufferWithCopy(pixels, 16);
  ASSERT_TRUE(buffer);

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);
  auto blit_pass = command_buffer->CreateBlitPass();
  ASSERT_TRUE(blit_pass);

  EXPECT_TRUE(blit_pass->AddCopy(BufferView(buffer, Range(0, 16)), texture,
                                 IRect::MakeXYWH(0, 0, 2, 2),
                                 "Deferred texture overwrite"));
  EXPECT_TRUE(blit_pass->EncodeCommands());
  EXPECT_TRUE(context_base->GetCommandQueue()->Submit({command_buffer}).ok());
  EXPECT_FALSE(texture_generated);
  EXPECT_FALSE(texture_uploaded);

  worker->SetAllowed(true);
  EXPECT_TRUE(context->GetReactor()->React());
  EXPECT_TRUE(texture_generated);
  EXPECT_TRUE(texture_uploaded);
}

namespace {

// Reports a fence as unsignaled [timeouts] times, then as [final_result].
class FenceMockGLESImpl : public NiceMock<MockGLESImpl> {
 public:
  FenceMockGLESImpl(int timeouts, GLenum final_result)
      : timeouts_(timeouts), final_result_(final_result) {}

  GLenum ClientWaitSync(GLsync sync,
                        GLbitfield flags,
                        GLuint64 timeout) override {
    waits_.push_back(flags);
    if (timeouts_ > 0) {
      timeouts_--;
      return GL_TIMEOUT_EXPIRED;
    }
    return final_result_;
  }

  const std::vector<GLbitfield>& GetWaits() const { return waits_; }

 private:
  int timeouts_;
  GLenum final_result_;
  std::vector<GLbitfield> waits_;
};

}  // namespace

TEST(CommandBufferGLES, CompletionCallbackWaitsForTheGPU) {
  auto impl = std::make_unique<FenceMockGLESImpl>(/*timeouts=*/2,
                                                  GL_CONDITION_SATISFIED);
  FenceMockGLESImpl* impl_ptr = impl.get();
  auto mock_gles = MockGLES::Init(std::move(impl));
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(true);
  context->AddReactorWorker(worker);

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);
  CommandBuffer::Status callback_status = CommandBuffer::Status::kPending;
  ASSERT_TRUE(context_base->GetCommandQueue()
                  ->Submit({command_buffer},
                           [&](CommandBuffer::Status status) {
                             callback_status = status;
                           })
                  .ok());

  EXPECT_EQ(callback_status, CommandBuffer::Status::kCompleted);
  // The first wait flushes so the fence is guaranteed to signal.
  ASSERT_EQ(impl_ptr->GetWaits().size(), 3u);
  EXPECT_EQ(impl_ptr->GetWaits()[0],
            static_cast<GLbitfield>(GL_SYNC_FLUSH_COMMANDS_BIT));
  EXPECT_EQ(impl_ptr->GetWaits()[1], 0u);
}

TEST(CommandBufferGLES, CompletionCallbackReportsAFailedGPUWait) {
  auto impl =
      std::make_unique<FenceMockGLESImpl>(/*timeouts=*/0, GL_WAIT_FAILED);
  auto mock_gles = MockGLES::Init(std::move(impl));
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(true);
  context->AddReactorWorker(worker);

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);
  CommandBuffer::Status callback_status = CommandBuffer::Status::kPending;
  (void)context_base->GetCommandQueue()->Submit(
      {command_buffer},
      [&](CommandBuffer::Status status) { callback_status = status; });

  EXPECT_EQ(callback_status, CommandBuffer::Status::kError);
}

TEST(CommandBufferGLES, SubmitWithoutCallbackDoesNotWaitForTheGPU) {
  auto impl =
      std::make_unique<FenceMockGLESImpl>(/*timeouts=*/0, GL_ALREADY_SIGNALED);
  FenceMockGLESImpl* impl_ptr = impl.get();
  auto mock_gles = MockGLES::Init(std::move(impl));
  auto context = CreateFakeGLESContext();
  auto context_base = std::static_pointer_cast<Context>(context);
  auto worker = std::make_shared<ToggleWorker>(true);
  context->AddReactorWorker(worker);

  auto command_buffer = context_base->CreateCommandBuffer();
  ASSERT_TRUE(command_buffer);
  ASSERT_TRUE(context_base->GetCommandQueue()->Submit({command_buffer}).ok());
  EXPECT_TRUE(impl_ptr->GetWaits().empty());
}

}  // namespace testing
}  // namespace impeller
