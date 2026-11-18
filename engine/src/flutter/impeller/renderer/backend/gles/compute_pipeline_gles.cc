// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "impeller/renderer/backend/gles/compute_pipeline_gles.h"

#include <utility>

namespace impeller {

ComputePipelineGLES::ComputePipelineGLES(
    std::shared_ptr<ReactorGLES> reactor,
    std::weak_ptr<PipelineLibrary> library,
    const ComputePipelineDescriptor& desc,
    std::shared_ptr<UniqueHandleGLES> handle,
    std::unique_ptr<BufferBindingsGLES> buffer_bindings)
    : Pipeline(std::move(library), desc),
      reactor_(std::move(reactor)),
      handle_(std::move(handle)),
      buffer_bindings_(std::move(buffer_bindings)),
      is_valid_(handle_ && handle_->IsValid() && buffer_bindings_) {
  if (is_valid_) {
    reactor_->SetDebugLabel(handle_->Get(), GetDescriptor().GetLabel());
  }
}

ComputePipelineGLES::~ComputePipelineGLES() = default;

std::shared_ptr<ComputePipelineGLES> ComputePipelineGLES::Create(
    std::shared_ptr<ReactorGLES> reactor,
    std::weak_ptr<PipelineLibrary> library,
    const ComputePipelineDescriptor& desc,
    std::shared_ptr<UniqueHandleGLES> handle,
    std::unique_ptr<BufferBindingsGLES> buffer_bindings) {
  auto pipeline = std::shared_ptr<ComputePipelineGLES>(new ComputePipelineGLES(
      std::move(reactor), std::move(library), desc, std::move(handle),
      std::move(buffer_bindings)));
  if (!pipeline->IsValid()) {
    return nullptr;
  }
  return pipeline;
}

bool ComputePipelineGLES::IsValid() const {
  return is_valid_;
}

const HandleGLES& ComputePipelineGLES::GetProgramHandle() const {
  return handle_->Get();
}

BufferBindingsGLES* ComputePipelineGLES::GetBufferBindings() const {
  return buffer_bindings_.get();
}

bool ComputePipelineGLES::BindProgram() const {
  if (!is_valid_) {
    return false;
  }
  std::optional<GLuint> program = reactor_->GetGLHandle(handle_->Get());
  if (!program.has_value()) {
    return false;
  }
  reactor_->GetProcTable().UseProgram(program.value());
  return true;
}

}  // namespace impeller
