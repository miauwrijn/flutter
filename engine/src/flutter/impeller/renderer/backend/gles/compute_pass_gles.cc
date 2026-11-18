// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "impeller/renderer/backend/gles/compute_pass_gles.h"

#include <utility>

#include "flutter/fml/closure.h"
#include "flutter/fml/trace_event.h"
#include "impeller/base/validation.h"
#include "impeller/renderer/backend/gles/buffer_bindings_gles.h"
#include "impeller/renderer/backend/gles/compute_pipeline_gles.h"
#include "impeller/renderer/command.h"

namespace impeller {

ComputePassGLES::ComputePassGLES(std::shared_ptr<const Context> context,
                                 std::shared_ptr<ReactorGLES> reactor)
    : ComputePass(std::move(context)), reactor_(std::move(reactor)) {}

ComputePassGLES::~ComputePassGLES() = default;

bool ComputePassGLES::IsValid() const {
  return reactor_ && reactor_->IsValid();
}

void ComputePassGLES::OnSetLabel(const std::string& label) {
  label_ = label;
}

void ComputePassGLES::SetCommandLabel(std::string_view label) {}

void ComputePassGLES::SetPipeline(
    const std::shared_ptr<Pipeline<ComputePipelineDescriptor>>& pipeline) {
  pipeline_ = pipeline;
}

bool ComputePassGLES::BindResource(ShaderStage stage,
                                   DescriptorType type,
                                   const ShaderUniformSlot& slot,
                                   const ShaderMetadata* metadata,
                                   BufferView view) {
  if (stage != ShaderStage::kCompute || !metadata || !view) {
    return false;
  }
  if (type != DescriptorType::kUniformBuffer &&
      type != DescriptorType::kStorageBuffer) {
    return false;
  }
  for (BufferBinding& binding : buffers_) {
    if (binding.metadata == metadata) {
      binding.type = type;
      binding.view = std::move(view);
      return true;
    }
  }
  buffers_.push_back(BufferBinding{
      .type = type, .metadata = metadata, .view = std::move(view)});
  return true;
}

bool ComputePassGLES::BindResource(ShaderStage stage,
                                   DescriptorType type,
                                   const SampledImageSlot& slot,
                                   const ShaderMetadata* metadata,
                                   std::shared_ptr<const Texture> texture,
                                   raw_ptr<const Sampler> sampler) {
  if (stage != ShaderStage::kCompute || !metadata || !texture || !sampler) {
    return false;
  }
  if (type != DescriptorType::kSampledImage) {
    return false;
  }
  for (TextureBinding& binding : textures_) {
    if (binding.metadata == metadata) {
      binding.slot = slot;
      binding.texture = std::move(texture);
      binding.sampler = sampler;
      return true;
    }
  }
  textures_.push_back(TextureBinding{.slot = slot,
                                     .metadata = metadata,
                                     .texture = std::move(texture),
                                     .sampler = sampler});
  return true;
}

absl::Status ComputePassGLES::Compute(std::array<uint32_t, 3> workgroup_count) {
  if (!pipeline_) {
    return absl::FailedPreconditionError("No compute pipeline is bound.");
  }
  if (workgroup_count[0] == 0u || workgroup_count[1] == 0u ||
      workgroup_count[2] == 0u) {
    return absl::OkStatus();
  }
  dispatches_.push_back(Dispatch{.pipeline = pipeline_,
                                 .buffers = buffers_,
                                 .textures = textures_,
                                 .workgroup_count = workgroup_count});
  return absl::OkStatus();
}

// Every dispatch is followed by a full memory barrier, which already orders
// what explicit barriers would.
void ComputePassGLES::AddBufferMemoryBarrier() {}

void ComputePassGLES::AddTextureMemoryBarrier() {}

bool ComputePassGLES::EncodeCommands() const {
  if (!IsValid()) {
    return false;
  }
  if (dispatches_.empty()) {
    return true;
  }
  return reactor_->AddOperation(
      [pass = shared_from_this()](const ReactorGLES& reactor) {
        if (!EncodeDispatchesInReactor(reactor, pass->dispatches_,
                                       pass->label_)) {
          VALIDATION_LOG << "Could not encode the compute pass '"
                         << pass->label_ << "'.";
        }
      },
      /*defer=*/true);
}

bool ComputePassGLES::EncodeDispatchesInReactor(
    const ReactorGLES& reactor,
    const std::vector<Dispatch>& dispatches,
    const std::string& label) {
  TRACE_EVENT0("impeller", "ComputePassGLES::EncodeDispatchesInReactor");
  const ProcTableGLES& gl = reactor.GetProcTable();
  fml::ScopedCleanupClosure pop_pass_debug_marker(
      [&gl]() { gl.PopDebugGroup(); });
  if (!label.empty()) {
    gl.PushDebugGroup(label);
  } else {
    pop_pass_debug_marker.Release();
  }

  SamplerStateCache sampler_cache;
  bool result = true;
  for (const Dispatch& dispatch : dispatches) {
    const ComputePipelineGLES& pipeline =
        ComputePipelineGLES::Cast(*dispatch.pipeline);
    if (!pipeline.BindProgram()) {
      VALIDATION_LOG << "Could not bind the compute program.";
      result = false;
      break;
    }
    BufferBindingsGLES* bindings = pipeline.GetBufferBindings();

    std::vector<BufferResource> uniform_buffers;
    for (const BufferBinding& binding : dispatch.buffers) {
      if (binding.type == DescriptorType::kStorageBuffer) {
        if (!bindings->BindStorageBuffer(
                gl, BufferResource(binding.metadata, binding.view))) {
          result = false;
          break;
        }
      } else {
        uniform_buffers.emplace_back(binding.metadata, binding.view);
      }
    }
    if (!result) {
      break;
    }

    std::vector<TextureAndSampler> textures;
    textures.reserve(dispatch.textures.size());
    for (const TextureBinding& binding : dispatch.textures) {
      textures.push_back(TextureAndSampler{
          .slot = binding.slot,
          .stage = ShaderStage::kCompute,
          .texture = TextureResource(binding.metadata, binding.texture),
          .sampler = binding.sampler,
      });
    }
    if (!bindings->BindComputeUniformData(gl, textures, uniform_buffers,
                                          &sampler_cache)) {
      VALIDATION_LOG << "Could not bind the compute shader resources.";
      result = false;
      break;
    }

    gl.DispatchCompute(dispatch.workgroup_count[0], dispatch.workgroup_count[1],
                       dispatch.workgroup_count[2]);
    // Make the dispatch's writes visible to the next dispatch and to every
    // later consumer: vertex, index, uniform and indirect reads, texture
    // fetches, copies and host reads through mapped buffers.
    gl.MemoryBarrier(GL_ALL_BARRIER_BITS);
  }
  gl.UseProgram(0u);
  return result;
}

}  // namespace impeller
