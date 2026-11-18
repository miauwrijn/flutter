# Compute in Flutter GPU

Flutter GPU runs compute shaders: GLSL programs that read and write buffers
and textures on the GPU, outside of any draw. Their output can feed later
render passes directly (particles, skinning, culling, simulation) or be read
back to the CPU.

This page covers what compute needs from the device, how to write and bundle a
compute shader, and the guarantees the API makes. The umbrella issue is
[#188474](https://github.com/flutter/flutter/issues/188474).

## Availability

| Platform | Backend | Compute |
| --- | --- | --- |
| iOS, macOS | Metal | Yes |
| Android | Vulkan | Yes |
| Android, OpenGL ES fallback devices | OpenGL ES 3.1+ | Yes |
| Android, OpenGL ES 3.0 devices | OpenGL ES 3.0 | No |
| Linux | OpenGL ES 3.1+ | Yes |
| Custom embedders | Desktop OpenGL 4.3+ | Yes |
| Windows | ANGLE on Direct3D 11 (OpenGL ES 3.0) | No |
| Web | none | No |

Check `gpu.gpuContext.supportsCompute` at runtime, and take your own non-compute
path where it is false. Flutter GPU never emulates compute: where it is
unsupported, `createComputePipeline` and `CommandBuffer.createComputePass`
throw an `UnsupportedError`.

## Writing a compute shader

```glsl
// particles.comp
layout(local_size_x = 64) in;

layout(std430) buffer;

struct Particle {
  vec4 color;
  vec2 position;
  vec2 velocity;
};

layout(binding = 0) buffer Particles {
  Particle particles[];
}
particles;

layout(binding = 1) uniform Step {
  float delta_seconds;
  uint count;
}
step;

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= step.count) {
    return;
  }
  particles.particles[i].position +=
      particles.particles[i].velocity * step.delta_seconds;
}
```

- **Workgroup size.** Declare it with a literal `local_size`. It is recorded
  in the shader bundle and used on every backend. Specialization constants
  (`local_size_x_id`) are rejected by the shader compiler.
- **Storage buffers** are `buffer` blocks. Mark them `readonly` or `writeonly`
  where you can; the access is reflected into the bundle.
- **Storage textures** are `image2D` uniforms with a format qualifier:
  `layout(binding = 0, rgba8) uniform writeonly image2D output_image;`. The
  formats every backend can bind are `rgba8`, `rgba16f`, `rgba32f` and `r32f`.
  OpenGL ES only allows read-write (neither `readonly` nor `writeonly`) access
  to `r32f` images.
- **Uniform blocks and sampled textures** work as they do in render shaders.

Add the shader to your bundle manifest with `"type": "compute"`:

```json
{
  "Particles": { "type": "compute", "file": "shaders/particles.comp" }
}
```

## Dispatching

```dart
final gpu.ShaderLibrary library =
    (await gpu.ShaderLibrary.fromAsset('build/shaderbundles/my.shaderbundle'))!;
final gpu.Shader shader = library['Particles']!;
final gpu.ComputePipeline pipeline =
    gpu.gpuContext.createComputePipeline(shader);

final gpu.CommandBuffer commandBuffer = gpu.gpuContext.createCommandBuffer();
final gpu.ComputePass pass = commandBuffer.createComputePass();
pass.bindPipeline(pipeline);
pass.bindStorageBuffer(shader.getStorageBufferSlot('Particles'), particles);
pass.bindUniform(shader.getUniformSlot('Step'), stepUniforms);
pass.dispatch((count + 63) ~/ 64);
commandBuffer.submit();
```

- `dispatch(x, [y, z])` takes **workgroup counts**, not invocations. A count
  of zero dispatches nothing.
- Bindings persist across dispatches in a pass, so a dispatch only rebinds what
  changed. Every resource the shader declares must be bound, even one it never
  uses, so that a pass that dispatches on one backend dispatches on all of
  them.
- In compute passes, storage buffers can also be bound through `BindingSet`s.
- `dispatchIndirect(arguments)` reads three `uint32` workgroup counts from a
  buffer, which an earlier dispatch can write.

### Limits

`gpuContext.maxComputeWorkgroupInvocations`, `maxComputeWorkgroupSize`,
`maxComputeWorkgroupCount`, `maxComputeSharedMemorySizeInBytes` and
`minimumStorageBufferByteAlignment` report the device's limits. The portable
floor is 128 invocations per workgroup, a 128 x 128 x 64 workgroup size and
16 KB of shared memory. A pipeline whose workgroup is larger than the device
allows fails to build, on every backend.

## Synchronization

Within a command buffer, and across command buffers on the queue, a write by
an earlier dispatch or pass is visible to every later dispatch, draw, copy and
readback, without a barrier. That includes indirect arguments, vertex and
index data, uniforms, sampled textures and storage textures.

A command buffer records one pass at a time, and passes run in the order they
are created. Creating the next pass, recording a copy or a readback, or
submitting ends the current pass. `end()` ends it explicitly, and any later
use of an ended pass throws a `StateError` naming what ended it.

## Using compute output

- **As vertex or instance data.** Bind the storage buffer as a vertex buffer of
  a later render pass; no copy is needed. See
  `A compute pass writes particles that an instanced draw renders` in
  `engine/src/flutter/testing/dart/gpu_test.dart`.
- **From vertex and fragment shaders.** `RenderPass.bindStorageBuffer` binds a
  storage buffer to a render shader, for example to read positions without a
  vertex layout.
- **As draw arguments.** `RenderPass.drawIndirect` and `drawIndexedIndirect`
  read the draw's counts from a buffer a compute shader wrote, so the GPU
  decides how much to draw.
- **As textures.** A storage texture written by a compute pass can be sampled
  by a later render pass.
- **On the CPU.** `CommandBuffer.readBuffer(view)` and
  `CommandBuffer.readTexture(region)` return a `Future<ByteData>` that
  completes once the command buffer finishes on the GPU. A read is recorded as
  a command, so it sees exactly the writes recorded before it. The bytes are
  tightly packed and owned by Dart. The future completes with an error if the
  submission fails or the command buffer is discarded without being submitted.

## OpenGL ES

Compute needs OpenGL ES 3.1 (or desktop OpenGL 4.3). The shader compiler
compiles each shader to the lowest GLSL version its features allow: compute
shaders, and vertex and fragment shaders that use storage buffers or images,
need GLSL ES 3.10. Everything else is unchanged.

OpenGL ES only links vertex and fragment shaders of the same GLSL version, so
in a bundle with storage in a render stage, every render shader is compiled to
3.10 and needs OpenGL ES 3.1. Keep such shaders in a bundle of their own so
your other render shaders still run on OpenGL ES 3.0 devices.

Looking up a shader that the device cannot run, `library['Name']`, throws an
`UnsupportedError` naming the shader, the OpenGL ES version it needs and the
version the device has. OpenGL ES 3.1 also allows devices to support no storage
blocks in vertex and fragment shaders; such a shader is reported the same way.

`supportsTextureFormat(format, shaderWrite: true)` says whether a format can be
bound as a storage texture on the current device.

## Hot reload

Compute shaders hot reload like render shaders. A pipeline built from a
reloaded shader is rebuilt on its next dispatch.
