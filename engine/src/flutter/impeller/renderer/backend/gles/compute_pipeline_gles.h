// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_COMPUTE_PIPELINE_GLES_H_
#define FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_COMPUTE_PIPELINE_GLES_H_

#include <memory>

#include "impeller/base/backend_cast.h"
#include "impeller/renderer/backend/gles/buffer_bindings_gles.h"
#include "impeller/renderer/backend/gles/reactor_gles.h"
#include "impeller/renderer/backend/gles/unique_handle_gles.h"
#include "impeller/renderer/compute_pipeline_descriptor.h"
#include "impeller/renderer/pipeline.h"

namespace impeller {

class PipelineLibraryGLES;

/// A linked program holding a single compute shader, and the bindings read
/// from it. Created on the reactor by `PipelineLibraryGLES`.
class ComputePipelineGLES final
    : public Pipeline<ComputePipelineDescriptor>,
      public BackendCast<ComputePipelineGLES,
                         Pipeline<ComputePipelineDescriptor>> {
 public:
  // |Pipeline|
  ~ComputePipelineGLES() override;

  const HandleGLES& GetProgramHandle() const;

  /// Uniform blocks, sampler uniforms and storage blocks of the program.
  BufferBindingsGLES* GetBufferBindings() const;

  [[nodiscard]] bool BindProgram() const;

  /// Wraps a linked compute program. Returns null if it is invalid.
  static std::shared_ptr<ComputePipelineGLES> Create(
      std::shared_ptr<ReactorGLES> reactor,
      std::weak_ptr<PipelineLibrary> library,
      const ComputePipelineDescriptor& desc,
      std::shared_ptr<UniqueHandleGLES> handle,
      std::unique_ptr<BufferBindingsGLES> buffer_bindings);

 private:
  std::shared_ptr<ReactorGLES> reactor_;
  std::shared_ptr<UniqueHandleGLES> handle_;
  std::unique_ptr<BufferBindingsGLES> buffer_bindings_;
  bool is_valid_ = false;

  ComputePipelineGLES(std::shared_ptr<ReactorGLES> reactor,
                      std::weak_ptr<PipelineLibrary> library,
                      const ComputePipelineDescriptor& desc,
                      std::shared_ptr<UniqueHandleGLES> handle,
                      std::unique_ptr<BufferBindingsGLES> buffer_bindings);

  // |Pipeline|
  bool IsValid() const override;

  ComputePipelineGLES(const ComputePipelineGLES&) = delete;

  ComputePipelineGLES& operator=(const ComputePipelineGLES&) = delete;
};

}  // namespace impeller

#endif  // FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_COMPUTE_PIPELINE_GLES_H_
