// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_COMPUTE_PASS_GLES_H_
#define FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_COMPUTE_PASS_GLES_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "flutter/fml/logging.h"
#include "impeller/core/buffer_view.h"
#include "impeller/core/raw_ptr.h"
#include "impeller/core/shader_types.h"
#include "impeller/renderer/backend/gles/reactor_gles.h"
#include "impeller/renderer/compute_pass.h"

namespace impeller {

/// A compute pass on OpenGL ES 3.1 or OpenGL 4.3.
///
/// Like `RenderPassGLES`, it records dispatches and encodes them on the
/// reactor when `EncodeCommands` is called. Each dispatch records the bindings
/// in effect when it is made, since bindings persist across dispatches.
///
/// Every dispatch is followed by `glMemoryBarrier(GL_ALL_BARRIER_BITS)`, so a
/// dispatch sees the writes of the dispatches before it, and every later
/// command, including host reads of storage buffers, sees the writes of the
/// pass.
class ComputePassGLES final
    : public ComputePass,
      public std::enable_shared_from_this<ComputePassGLES> {
 public:
  // |ComputePass|
  ~ComputePassGLES() override;

 private:
  friend class CommandBufferGLES;

  struct BufferBinding {
    DescriptorType type = DescriptorType::kUniformBuffer;
    const ShaderMetadata* metadata = nullptr;
    BufferView view;
  };

  struct TextureBinding {
    /// kSampledImage, or kStorageImage for a texture bound as an image.
    DescriptorType type = DescriptorType::kSampledImage;
    SampledImageSlot slot;
    const ShaderMetadata* metadata = nullptr;
    std::shared_ptr<const Texture> texture;
    raw_ptr<const Sampler> sampler;
  };

  struct Dispatch {
    std::shared_ptr<Pipeline<ComputePipelineDescriptor>> pipeline;
    std::vector<BufferBinding> buffers;
    std::vector<TextureBinding> textures;
    std::array<uint32_t, 3> workgroup_count = {0u, 0u, 0u};
  };

  std::shared_ptr<ReactorGLES> reactor_;
  std::string label_;
  std::shared_ptr<Pipeline<ComputePipelineDescriptor>> pipeline_;
  // The bindings in effect, one per shader binding.
  std::vector<BufferBinding> buffers_;
  std::vector<TextureBinding> textures_;
  std::vector<Dispatch> dispatches_;

  ComputePassGLES(std::shared_ptr<const Context> context,
                  std::shared_ptr<ReactorGLES> reactor);

  // |ComputePass|
  bool IsValid() const override;

  // |ComputePass|
  void OnSetLabel(const std::string& label) override;

  // |ComputePass|
  void SetCommandLabel(std::string_view label) override;

  // |ComputePass|
  void SetPipeline(const std::shared_ptr<Pipeline<ComputePipelineDescriptor>>&
                       pipeline) override;

  // |ComputePass|
  absl::Status Compute(std::array<uint32_t, 3> workgroup_count) override;

  // |ResourceBinder|
  bool BindResource(ShaderStage stage,
                    DescriptorType type,
                    const ShaderUniformSlot& slot,
                    const ShaderMetadata* metadata,
                    BufferView view) override;

  // |ResourceBinder|
  bool BindResource(ShaderStage stage,
                    DescriptorType type,
                    const SampledImageSlot& slot,
                    const ShaderMetadata* metadata,
                    std::shared_ptr<const Texture> texture,
                    raw_ptr<const Sampler> sampler) override;

  // |ComputePass|
  void AddBufferMemoryBarrier() override;

  // |ComputePass|
  void AddTextureMemoryBarrier() override;

  // |ComputePass|
  bool EncodeCommands() const override;

  static bool BindStorageTexture(const ProcTableGLES& gl,
                                 const TextureBinding& binding);

  static bool EncodeDispatchesInReactor(const ReactorGLES& reactor,
                                        const std::vector<Dispatch>& dispatches,
                                        const std::string& label);

  ComputePassGLES(const ComputePassGLES&) = delete;

  ComputePassGLES& operator=(const ComputePassGLES&) = delete;
};

}  // namespace impeller

#endif  // FLUTTER_IMPELLER_RENDERER_BACKEND_GLES_COMPUTE_PASS_GLES_H_
