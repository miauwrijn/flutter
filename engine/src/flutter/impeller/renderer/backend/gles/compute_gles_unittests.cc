// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <array>
#include <cstring>
#include <memory>

#include "flutter/testing/testing.h"  // IWYU pragma: keep
#include "gtest/gtest.h"
#include "impeller/core/shader_types.h"
#include "impeller/renderer/backend/gles/buffer_bindings_gles.h"
#include "impeller/renderer/backend/gles/capabilities_gles.h"
#include "impeller/renderer/backend/gles/device_buffer_gles.h"
#include "impeller/renderer/backend/gles/reactor_gles.h"
#include "impeller/renderer/backend/gles/test/mock_gles.h"
#include "impeller/renderer/command.h"

namespace impeller {
namespace testing {

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace {

class TestWorker : public ReactorGLES::Worker {
 public:
  bool CanReactorReactOnCurrentThreadNow(
      const ReactorGLES& reactor) const override {
    return true;
  }
};

std::shared_ptr<ReactorGLES> CreateReactor() {
  // The reactor holds its workers weakly, so keep this one alive.
  static const std::shared_ptr<TestWorker> kWorker =
      std::make_shared<TestWorker>();
  auto proc_table = std::make_unique<ProcTableGLES>(kMockResolverGLES);
  auto reactor = std::make_shared<ReactorGLES>(std::move(proc_table));
  reactor->AddWorker(kWorker);
  return reactor;
}

std::unique_ptr<DeviceBufferGLES> CreateBuffer(
    const std::shared_ptr<ReactorGLES>& reactor,
    size_t size) {
  auto backing_store = std::make_unique<Allocation>();
  if (!backing_store->Truncate(Bytes{size})) {
    return nullptr;
  }
  return std::make_unique<DeviceBufferGLES>(
      DeviceBufferDescriptor{.storage_mode = StorageMode::kHostVisible,
                             .size = size},
      reactor, std::move(backing_store));
}

// Reports compute limits that satisfy the OpenGL ES 3.1 minimums.
void ExpectComputeLimitQueries(MockGLESImpl& impl) {
  ON_CALL(impl, GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, _, _))
      .WillByDefault([](GLenum, GLuint index, GLint* data) {
        *data = 65535 - static_cast<GLint>(index);
      });
  ON_CALL(impl, GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, _, _))
      .WillByDefault([](GLenum, GLuint index, GLint* data) {
        *data = index == 2 ? 64 : 128;
      });
  ON_CALL(impl, GetIntegerv(_, _)).WillByDefault([](GLenum name, GLint* value) {
    switch (name) {
      case GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS:
        *value = 128;
        break;
      case GL_MAX_COMPUTE_SHARED_MEMORY_SIZE:
        *value = 16384;
        break;
      case GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS:
        *value = 4;
        break;
      case GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT:
        *value = 512;
        break;
      default:
        break;
    }
  });
}

}  // namespace

TEST(ComputeGLESTest, CapabilitiesReportComputeOnOpenGLES31) {
  auto impl = std::make_unique<NiceMock<MockGLESImpl>>();
  ExpectComputeLimitQueries(*impl);
  std::shared_ptr<MockGLES> mock_gl =
      MockGLES::Init(std::move(impl), std::nullopt, "OpenGL ES 3.1");
  // The capabilities of `mock_gl`'s own proc table are read before the mock
  // is registered, so read them from a proc table created afterwards.
  ProcTableGLES proc_table(kMockResolverGLES);
  const std::shared_ptr<const CapabilitiesGLES>& capabilities =
      proc_table.GetCapabilities();

  ASSERT_TRUE(capabilities->SupportsCompute());
  const ComputeLimits limits = capabilities->GetComputeLimits();
  EXPECT_EQ(limits.max_workgroup_invocations, 128u);
  EXPECT_EQ(limits.max_workgroup_size,
            (std::array<uint32_t, 3>{128u, 128u, 64u}));
  EXPECT_EQ(limits.max_workgroup_count,
            (std::array<uint32_t, 3>{65535u, 65534u, 65533u}));
  EXPECT_EQ(limits.max_shared_memory_size, 16384u);
  EXPECT_EQ(capabilities->GetMaxShaderStorageBlocks(ShaderStage::kCompute), 4u);
  // The larger of the uniform alignment (256) and the storage alignment.
  EXPECT_EQ(capabilities->GetMinimumStorageBufferAlignment(), 512u);
}

TEST(ComputeGLESTest, CapabilitiesReportNoComputeOnOpenGLES30) {
  auto impl = std::make_unique<NiceMock<MockGLESImpl>>();
  ExpectComputeLimitQueries(*impl);
  std::shared_ptr<MockGLES> mock_gl =
      MockGLES::Init(std::move(impl), std::nullopt, "OpenGL ES 3.0");
  // The capabilities of `mock_gl`'s own proc table are read before the mock
  // is registered, so read them from a proc table created afterwards.
  ProcTableGLES proc_table(kMockResolverGLES);
  const std::shared_ptr<const CapabilitiesGLES>& capabilities =
      proc_table.GetCapabilities();

  EXPECT_FALSE(capabilities->SupportsCompute());
  const ComputeLimits limits = capabilities->GetComputeLimits();
  EXPECT_EQ(limits.max_workgroup_invocations, 0u);
  EXPECT_EQ(limits.max_workgroup_size, (std::array<uint32_t, 3>{0u, 0u, 0u}));
}

TEST(ComputeGLESTest, StorageBuffersBindToTheShaderBindingByName) {
  auto impl = std::make_unique<NiceMock<MockGLESImpl>>();
  ON_CALL(*impl, GetProgramInterfaceiv(_, GL_SHADER_STORAGE_BLOCK,
                                       GL_ACTIVE_RESOURCES, _))
      .WillByDefault(
          [](GLuint, GLenum, GLenum, GLint* params) { *params = 1; });
  ON_CALL(*impl, GetProgramInterfaceiv(_, GL_SHADER_STORAGE_BLOCK,
                                       GL_MAX_NAME_LENGTH, _))
      .WillByDefault(
          [](GLuint, GLenum, GLenum, GLint* params) { *params = 16; });
  ON_CALL(*impl, GetProgramResourceName(_, GL_SHADER_STORAGE_BLOCK, 0, _, _, _))
      .WillByDefault(
          [](GLuint, GLenum, GLuint, GLsizei, GLsizei* length, GLchar* name) {
            std::strcpy(name, "Vertices");
            *length = 8;
          });
  ON_CALL(*impl,
          GetProgramResourceiv(_, GL_SHADER_STORAGE_BLOCK, 0, 1, _, 1, _, _))
      .WillByDefault([](GLuint, GLenum, GLuint, GLsizei, const GLenum* props,
                        GLsizei, GLsizei*, GLint* params) {
        EXPECT_EQ(props[0], static_cast<GLenum>(GL_BUFFER_BINDING));
        *params = 3;
      });
  // Only the declared storage block binds, at the shader's binding point.
  EXPECT_CALL(*impl, BindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, _, 256, 16))
      .Times(1);
  std::shared_ptr<MockGLES> mock_gl =
      MockGLES::Init(std::move(impl), std::nullopt, "OpenGL ES 3.1");
  std::shared_ptr<ReactorGLES> reactor = CreateReactor();

  BufferBindingsGLES bindings;
  ASSERT_TRUE(bindings.ReadStorageBlockBindings(mock_gl->GetProcTable(),
                                                /*program=*/1));

  std::unique_ptr<DeviceBufferGLES> buffer = CreateBuffer(reactor, 512);
  ASSERT_TRUE(buffer);
  ShaderMetadata vertices_metadata = {.name = "Vertices"};
  ShaderMetadata unused_metadata = {.name = "Unused"};
  EXPECT_FALSE(buffer->IsWrittenByGPU());
  EXPECT_TRUE(bindings.BindStorageBuffer(
      mock_gl->GetProcTable(),
      BufferResource(&vertices_metadata,
                     BufferView(buffer.get(), Range{256, 16}))));
  // A storage binding may be written by the shader.
  EXPECT_TRUE(buffer->IsWrittenByGPU());
  // A block the driver removed is skipped.
  EXPECT_TRUE(bindings.BindStorageBuffer(
      mock_gl->GetProcTable(),
      BufferResource(&unused_metadata,
                     BufferView(buffer.get(), Range{0, 16}))));
}

TEST(ComputeGLESTest, GPUWrittenBuffersReadBackIntoTheShadowCopy) {
  auto impl = std::make_unique<NiceMock<MockGLESImpl>>();
  std::array<uint8_t, 8> gpu_contents = {10, 11, 12, 13, 14, 15, 16, 17};
  {
    InSequence sequence;
    // The first bind uploads the initial host contents.
    EXPECT_CALL(*impl, BufferSubData(_, 0, 8, _)).Times(1);
    // A host write made while the buffer is GPU-written is uploaded before
    // the buffer is read back, so the read returns it.
    EXPECT_CALL(*impl, BufferSubData(GL_COPY_READ_BUFFER, 0, 2, _)).Times(1);
    EXPECT_CALL(*impl,
                MapBufferRange(GL_COPY_READ_BUFFER, 0, 8, GL_MAP_READ_BIT))
        .WillOnce(Return(gpu_contents.data()));
    EXPECT_CALL(*impl, UnmapBuffer(GL_COPY_READ_BUFFER))
        .WillOnce(Return(GL_TRUE));
  }
  std::shared_ptr<MockGLES> mock_gl =
      MockGLES::Init(std::move(impl), std::nullopt, "OpenGL ES 3.1");
  std::shared_ptr<ReactorGLES> reactor = CreateReactor();

  std::unique_ptr<DeviceBufferGLES> buffer = CreateBuffer(reactor, 8);
  ASSERT_TRUE(buffer);
  const std::array<uint8_t, 8> host_contents = {1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_TRUE(buffer->CopyHostBuffer(host_contents.data(), Range{0, 8}));
  ASSERT_TRUE(buffer->BindAndUploadDataIfNecessary(
      DeviceBufferGLES::BindingType::kShaderStorageBuffer));

  buffer->MarkWrittenByGPU();
  gpu_contents[0] = 20;
  gpu_contents[1] = 21;
  const std::array<uint8_t, 2> host_write = {20, 21};
  ASSERT_TRUE(buffer->CopyHostBuffer(host_write.data(), Range{0, 2}));

  const uint8_t* contents = buffer->GetBufferData();
  EXPECT_FALSE(buffer->IsWrittenByGPU());
  EXPECT_EQ(std::memcmp(contents, gpu_contents.data(), gpu_contents.size()), 0);
  // The shadow copy is current again, so a second read does not map.
  buffer->GetBufferData();
}

TEST(ComputeGLESTest, BuffersTheGPUNeverWroteAreNotReadBack) {
  auto impl = std::make_unique<NiceMock<MockGLESImpl>>();
  EXPECT_CALL(*impl, MapBufferRange(_, _, _, _)).Times(0);
  std::shared_ptr<MockGLES> mock_gl =
      MockGLES::Init(std::move(impl), std::nullopt, "OpenGL ES 3.1");
  std::shared_ptr<ReactorGLES> reactor = CreateReactor();

  std::unique_ptr<DeviceBufferGLES> buffer = CreateBuffer(reactor, 8);
  ASSERT_TRUE(buffer);
  ASSERT_TRUE(buffer->BindAndUploadDataIfNecessary(
      DeviceBufferGLES::BindingType::kArrayBuffer));
  buffer->GetBufferData();
  // A buffer that was never uploaded has no GL buffer to read back.
  std::unique_ptr<DeviceBufferGLES> never_uploaded = CreateBuffer(reactor, 8);
  never_uploaded->MarkWrittenByGPU();
  never_uploaded->GetBufferData();
  EXPECT_FALSE(never_uploaded->IsWrittenByGPU());
}

}  // namespace testing
}  // namespace impeller
