// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "impeller/renderer/backend/vulkan/compute_pass_vk.h"

#include <algorithm>

#include <vector>

#include "flutter/fml/closure.h"
#include "flutter/fml/logging.h"
#include "impeller/renderer/backend/vulkan/command_buffer_vk.h"
#include "impeller/renderer/backend/vulkan/compute_pipeline_vk.h"
#include "impeller/renderer/backend/vulkan/device_buffer_vk.h"
#include "impeller/renderer/backend/vulkan/formats_vk.h"
#include "impeller/renderer/backend/vulkan/sampler_vk.h"
#include "impeller/renderer/backend/vulkan/texture_vk.h"
#include "vulkan/vulkan_structs.hpp"

namespace impeller {

// The stages in which render and blit passes access memory.
static constexpr vk::PipelineStageFlags kGraphicsAndTransferStages =
    vk::PipelineStageFlagBits::eDrawIndirect |
    vk::PipelineStageFlagBits::eVertexInput |
    vk::PipelineStageFlagBits::eVertexShader |
    vk::PipelineStageFlagBits::eFragmentShader |
    vk::PipelineStageFlagBits::eEarlyFragmentTests |
    vk::PipelineStageFlagBits::eLateFragmentTests |
    vk::PipelineStageFlagBits::eColorAttachmentOutput |
    vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eHost;

// The writes render and blit passes and the host can make.
static constexpr vk::AccessFlags kGraphicsAndTransferWrites =
    vk::AccessFlagBits::eShaderWrite |
    vk::AccessFlagBits::eColorAttachmentWrite |
    vk::AccessFlagBits::eDepthStencilAttachmentWrite |
    vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eHostWrite;

// Records a global memory barrier.
//
// https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
// suggests that anything more finely grained than a global memory barrier is
// likely to be weakened into one.
static void RecordMemoryBarrier(const vk::CommandBuffer& command_buffer,
                                vk::PipelineStageFlags src_stage,
                                vk::AccessFlags src_access,
                                vk::PipelineStageFlags dst_stage,
                                vk::AccessFlags dst_access) {
  vk::MemoryBarrier barrier;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  command_buffer.pipelineBarrier(/*srcStageMask=*/src_stage,
                                 /*dstStageMask=*/dst_stage,
                                 /*dependencyFlags=*/{},
                                 /*memoryBarrierCount=*/1,
                                 /*pMemoryBarriers=*/&barrier,
                                 /*bufferMemoryBarrierCount=*/0,
                                 /*pBufferMemoryBarriers=*/nullptr,
                                 /*imageMemoryBarrierCount=*/0,
                                 /*pImageMemoryBarriers=*/nullptr);
}

ComputePassVK::ComputePassVK(std::shared_ptr<const Context> context,
                             std::shared_ptr<CommandBufferVK> command_buffer)
    : ComputePass(std::move(context)),
      command_buffer_(std::move(command_buffer)) {
  is_valid_ = true;
}

ComputePassVK::~ComputePassVK() = default;

bool ComputePassVK::IsValid() const {
  return is_valid_;
}

void ComputePassVK::OnSetLabel(const std::string& label) {
  if (label.empty()) {
    return;
  }
  label_ = label;
}

// |RenderPass|
void ComputePassVK::SetCommandLabel(std::string_view label) {
#ifdef IMPELLER_DEBUG
  command_buffer_->PushDebugGroup(label);
  has_label_ = true;
#endif  // IMPELLER_DEBUG
}

void ComputePassVK::PopCommandLabel() {
#ifdef IMPELLER_DEBUG
  if (has_label_) {
    command_buffer_->PopDebugGroup();
  }
  has_label_ = false;
#endif  // IMPELLER_DEBUG
}

// |ComputePass|
void ComputePassVK::SetPipeline(
    const std::shared_ptr<Pipeline<ComputePipelineDescriptor>>& pipeline) {
  FML_DCHECK(pipeline) << "A compute pass must not be given a null pipeline.";
  const auto& pipeline_vk = ComputePipelineVK::Cast(*pipeline);
  command_buffer_->GetCommandBuffer().bindPipeline(
      vk::PipelineBindPoint::eCompute, pipeline_vk.GetPipeline());
  pipeline_ = pipeline;
  descriptor_set_dirty_ = true;
}

absl::Status ComputePassVK::BindDescriptorSet() {
  const ContextVK& context_vk = ContextVK::Cast(*context_);
  const auto& pipeline_vk = ComputePipelineVK::Cast(*pipeline_);

  auto descriptor_result = command_buffer_->AllocateDescriptorSets(
      pipeline_vk.GetDescriptorSetLayout(), pipeline_vk.GetPipelineKey(),
      context_vk);
  if (!descriptor_result.ok()) {
    return descriptor_result.status();
  }
  vk::DescriptorSet descriptor_set = descriptor_result.value();

  // Write every binding the pipeline declares that the pass has a resource
  // for. Bindings made for another pipeline in the pass are kept for later
  // dispatches but not written here.
  std::vector<vk::WriteDescriptorSet> writes;
  for (const DescriptorSetLayout& layout :
       pipeline_->GetDescriptor().GetDescriptorSetLayouts()) {
    auto found = bindings_.find(layout.binding);
    if (found == bindings_.end() ||
        found->second.type != layout.descriptor_type) {
      continue;
    }
    const BoundResource& resource = found->second;
    vk::WriteDescriptorSet write_set;
    write_set.dstSet = descriptor_set;
    write_set.dstBinding = layout.binding;
    write_set.descriptorCount = 1u;
    write_set.descriptorType = ToVKDescriptorType(resource.type);
    if (resource.image_info.has_value()) {
      write_set.pImageInfo = &resource.image_info.value();
    } else if (resource.buffer_info.has_value()) {
      write_set.pBufferInfo = &resource.buffer_info.value();
    } else {
      continue;
    }
    writes.push_back(write_set);
  }

  context_vk.GetDevice().updateDescriptorSets(writes.size(), writes.data(), 0u,
                                              {});
  command_buffer_->GetCommandBuffer().bindDescriptorSets(
      /*pipelineBindPoint=*/vk::PipelineBindPoint::eCompute,
      /*layout=*/pipeline_vk.GetPipelineLayout(),
      /*firstSet=*/0,
      /*descriptorSetCount=*/1,
      /*pDescriptorSets=*/&descriptor_set,
      /*dynamicOffsetCount=*/0,
      /*pDynamicOffsets=*/nullptr);
  descriptor_set_dirty_ = false;
  return absl::OkStatus();
}

// |ComputePass|
absl::Status ComputePassVK::Compute(std::array<uint32_t, 3> workgroup_count) {
  // The label set by `SetCommandLabel` applies to this dispatch only.
  fml::ScopedCleanupClosure pop_label([this]() { PopCommandLabel(); });

  // A dispatch with no workgroups does nothing, like a draw with no vertices.
  if (workgroup_count[0] == 0u || workgroup_count[1] == 0u ||
      workgroup_count[2] == 0u) {
    return absl::OkStatus();
  }

  if (!pipeline_) {
    return absl::Status(absl::StatusCode::kCancelled,
                        "No compute pipeline is bound.");
  }

  if (absl::Status status = PrepareDispatch(); !status.ok()) {
    return status;
  }

  // The arguments are workgroup counts. The per-workgroup invocation count (the
  // local size) is baked into the shader module, so dispatch the counts
  // directly.
  command_buffer_->GetCommandBuffer().dispatch(
      workgroup_count[0], workgroup_count[1], workgroup_count[2]);
  return absl::OkStatus();
}

// |ComputePass|
absl::Status ComputePassVK::ComputeIndirect(BufferView indirect_buffer) {
  fml::ScopedCleanupClosure pop_label([this]() { PopCommandLabel(); });

  if (!pipeline_) {
    return absl::Status(absl::StatusCode::kCancelled,
                        "No compute pipeline is bound.");
  }
  if (!indirect_buffer) {
    return absl::InvalidArgumentError("No indirect buffer.");
  }
  const vk::Buffer buffer =
      DeviceBufferVK::Cast(*indirect_buffer.GetBuffer()).GetBuffer();
  const vk::DeviceSize offset = indirect_buffer.GetRange().offset;
  std::shared_ptr<const DeviceBuffer> device_buffer =
      indirect_buffer.TakeBuffer();
  if (device_buffer && !command_buffer_->Track(device_buffer)) {
    return absl::InternalError("Could not track the indirect buffer.");
  }

  if (absl::Status status = PrepareDispatch(); !status.ok()) {
    return status;
  }
  // The counts are a VkDispatchIndirectCommand.
  command_buffer_->GetCommandBuffer().dispatchIndirect(buffer, offset);
  return absl::OkStatus();
}

absl::Status ComputePassVK::PrepareDispatch() {
  // Layout transitions are recorded outside of the dispatch, before the
  // memory barrier below, which orders the dispatch after them.
  if (!TransitionStorageTextures()) {
    return absl::InternalError(
        "Could not move a storage texture to the general layout.");
  }

  if (descriptor_set_dirty_) {
    if (absl::Status status = BindDescriptorSet(); !status.ok()) {
      return status;
    }
  }

  if (has_dispatched_) {
    // Make the writes of earlier dispatches in the pass visible to this one,
    // and order this dispatch's writes after their reads and writes.
    // Indirect dispatches read their counts in the draw indirect stage.
    RecordMemoryBarrier(command_buffer_->GetCommandBuffer(),
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::AccessFlagBits::eShaderWrite,
                        vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eDrawIndirect,
                        vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eIndirectCommandRead);
  } else {
    // Make the writes of earlier render and blit passes visible to the first
    // dispatch, and order its writes after their reads and writes. Recording
    // this here, rather than in the render and blit passes, means only
    // command buffers that use compute pay for it. The shader-read layout
    // transitions those passes record complete before the fragment shader
    // stage, which is in this barrier's source scope, so the transitions are
    // ordered too.
    RecordMemoryBarrier(command_buffer_->GetCommandBuffer(),
                        kGraphicsAndTransferStages, kGraphicsAndTransferWrites,
                        vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eDrawIndirect,
                        vk::AccessFlagBits::eUniformRead |
                            vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eIndirectCommandRead);
  }
  has_dispatched_ = true;
  return absl::OkStatus();
}

// |ResourceBinder|
bool ComputePassVK::BindResource(ShaderStage stage,
                                 DescriptorType type,
                                 const ShaderUniformSlot& slot,
                                 const ShaderMetadata* metadata,
                                 BufferView view) {
  return BindResource(slot.binding, type, view);
}

// |ResourceBinder|
bool ComputePassVK::BindResource(ShaderStage stage,
                                 DescriptorType type,
                                 const SampledImageSlot& slot,
                                 const ShaderMetadata* metadata,
                                 std::shared_ptr<const Texture> texture,
                                 raw_ptr<const Sampler> sampler) {
  if (!texture || !texture->IsValid()) {
    return false;
  }
  const TextureVK& texture_vk = TextureVK::Cast(*texture);

  if (!command_buffer_->Track(texture)) {
    return false;
  }

  if (type == DescriptorType::kStorageImage) {
    // Storage images are read and written without a sampler, in the general
    // layout. The view must not swizzle, so use the plain image view.
    vk::DescriptorImageInfo image_info;
    image_info.imageLayout = vk::ImageLayout::eGeneral;
    image_info.imageView = texture_vk.GetImageView();
    bindings_[static_cast<uint32_t>(slot.binding)] = {
        .type = type, .image_info = image_info, .storage_texture = texture};
    descriptor_set_dirty_ = true;
    return true;
  }

  if (!sampler) {
    return false;
  }
  const SamplerVK& sampler_vk = SamplerVK::Cast(*sampler);

  vk::DescriptorImageInfo image_info;
  image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  image_info.sampler = sampler_vk.GetSampler();
  image_info.imageView = texture_vk.GetSampledImageView();

  bindings_[static_cast<uint32_t>(slot.binding)] = {.type = type,
                                                    .image_info = image_info};
  descriptor_set_dirty_ = true;
  return true;
}

bool ComputePassVK::BindResource(size_t binding,
                                 DescriptorType type,
                                 BufferView view) {
  if (!view.GetBuffer()) {
    return false;
  }
  auto buffer = DeviceBufferVK::Cast(*view.GetBuffer()).GetBuffer();
  if (!buffer) {
    return false;
  }

  std::shared_ptr<const DeviceBuffer> device_buffer = view.TakeBuffer();
  if (device_buffer && !command_buffer_->Track(device_buffer)) {
    return false;
  }

  vk::DescriptorBufferInfo buffer_info;
  buffer_info.buffer = buffer;
  buffer_info.offset = view.GetRange().offset;
  buffer_info.range = view.GetRange().length;

  bindings_[static_cast<uint32_t>(binding)] = {.type = type,
                                               .buffer_info = buffer_info};
  descriptor_set_dirty_ = true;
  return true;
}

bool ComputePassVK::TransitionStorageTextures() {
  for (const DescriptorSetLayout& layout :
       pipeline_->GetDescriptor().GetDescriptorSetLayouts()) {
    if (layout.descriptor_type != DescriptorType::kStorageImage) {
      continue;
    }
    auto found = bindings_.find(layout.binding);
    if (found == bindings_.end() || !found->second.storage_texture) {
      continue;
    }
    const std::shared_ptr<const Texture>& texture =
        found->second.storage_texture;
    const TextureVK& texture_vk = TextureVK::Cast(*texture);
    if (texture_vk.GetLayout() == vk::ImageLayout::eGeneral) {
      continue;
    }
    BarrierVK barrier;
    barrier.cmd_buffer = command_buffer_->GetCommandBuffer();
    barrier.new_layout = vk::ImageLayout::eGeneral;
    // Wait for earlier render, blit and compute work on the image, and for
    // its earlier reads, before changing its layout.
    barrier.src_stage =
        kGraphicsAndTransferStages | vk::PipelineStageFlagBits::eComputeShader;
    barrier.src_access =
        kGraphicsAndTransferWrites | vk::AccessFlagBits::eShaderWrite;
    barrier.dst_stage = vk::PipelineStageFlagBits::eComputeShader;
    barrier.dst_access =
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    if (!texture_vk.SetLayout(barrier)) {
      return false;
    }
    if (std::find(storage_textures_.begin(), storage_textures_.end(),
                  texture) == storage_textures_.end()) {
      storage_textures_.push_back(texture);
    }
  }
  return true;
}

// |ComputePass|
void ComputePassVK::AddBufferMemoryBarrier() {
  RecordMemoryBarrier(command_buffer_->GetCommandBuffer(),
                      vk::PipelineStageFlagBits::eComputeShader,
                      vk::AccessFlagBits::eShaderWrite,
                      vk::PipelineStageFlagBits::eComputeShader,
                      vk::AccessFlagBits::eShaderRead);
}

// |ComputePass|
void ComputePassVK::AddTextureMemoryBarrier() {
  RecordMemoryBarrier(command_buffer_->GetCommandBuffer(),
                      vk::PipelineStageFlagBits::eComputeShader,
                      vk::AccessFlagBits::eShaderWrite,
                      vk::PipelineStageFlagBits::eComputeShader,
                      vk::AccessFlagBits::eShaderRead);
}

// |ComputePass|
bool ComputePassVK::EncodeCommands() const {
  // A pass that dispatched nothing has no writes to order.
  if (!has_dispatched_) {
    return true;
  }
  // Make the pass's writes visible to every later consumer in queue order,
  // whether in this command buffer or a later one: draws reading vertex,
  // index, indirect or uniform data, any shader stage, copies and host reads.
  // Order later writes after the pass's reads and writes too.
  //
  // This does not currently handle image layout transitions, since compute
  // passes do not write images.
  RecordMemoryBarrier(
      command_buffer_->GetCommandBuffer(),
      vk::PipelineStageFlagBits::eComputeShader,
      vk::AccessFlagBits::eShaderWrite,
      vk::PipelineStageFlagBits::eDrawIndirect |
          vk::PipelineStageFlagBits::eVertexInput |
          vk::PipelineStageFlagBits::eVertexShader |
          vk::PipelineStageFlagBits::eFragmentShader |
          vk::PipelineStageFlagBits::eComputeShader |
          vk::PipelineStageFlagBits::eTransfer |
          vk::PipelineStageFlagBits::eHost,
      vk::AccessFlagBits::eIndirectCommandRead |
          vk::AccessFlagBits::eIndexRead |
          vk::AccessFlagBits::eVertexAttributeRead |
          vk::AccessFlagBits::eUniformRead | vk::AccessFlagBits::eShaderRead |
          vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
          vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eHostRead);

  // Later passes sample textures, and copy from them, starting from the
  // shader-read layout. The barrier above already made the pass's writes
  // available, so these transitions only need to wait for the compute stage.
  for (const std::shared_ptr<const Texture>& texture : storage_textures_) {
    BarrierVK barrier;
    barrier.cmd_buffer = command_buffer_->GetCommandBuffer();
    barrier.new_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.src_stage = vk::PipelineStageFlagBits::eComputeShader;
    barrier.src_access = vk::AccessFlagBits::eShaderWrite;
    barrier.dst_stage = vk::PipelineStageFlagBits::eVertexShader |
                        vk::PipelineStageFlagBits::eFragmentShader |
                        vk::PipelineStageFlagBits::eComputeShader |
                        vk::PipelineStageFlagBits::eTransfer;
    barrier.dst_access =
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
    if (!TextureVK::Cast(*texture).SetLayout(barrier)) {
      return false;
    }
  }
  return true;
}

}  // namespace impeller
