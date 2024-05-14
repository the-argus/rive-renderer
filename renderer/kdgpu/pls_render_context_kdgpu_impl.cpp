#include "rive/pls/kdgpu/pls_render_context_kdgpu_impl.hpp"
#include "rive/pls/pls_image.hpp"
// #include "shaders/out/generated/spirv/color_ramp.frag.h"
// #include "shaders/out/generated/spirv/color_ramp.vert.h"
#include <KDGpu/adapter.h>
#include <KDGpu/bind_group_layout_options.h>
#include <KDGpu/bind_group_options.h>
#include <KDGpu/buffer.h>
#include <KDGpu/buffer_options.h>
#include <KDGpu/graphics_api.h>
#include <KDGpu/graphics_pipeline_options.h>
#include <KDGpu/sampler.h>
#include <KDGpu/texture_options.h>
#include <KDGpu/vulkan/vulkan_render_pass_command_recorder.h>
#include <KDGpu/vulkan/vulkan_resource_manager.h>

#include "shaders/constants.glsl"

#include "shaders/out/generated/spirv/color_ramp.frag.h"
#include "shaders/out/generated/spirv/color_ramp.vert.h"
#include "shaders/out/generated/spirv/draw_image_mesh.frag.h"
#include "shaders/out/generated/spirv/draw_image_mesh.vert.h"
#include "shaders/out/generated/spirv/draw_interior_triangles.frag.h"
#include "shaders/out/generated/spirv/draw_interior_triangles.vert.h"
#include "shaders/out/generated/spirv/draw_path.frag.h"
#include "shaders/out/generated/spirv/draw_path.vert.h"
#include "shaders/out/generated/spirv/tessellate.frag.h"
#include "shaders/out/generated/spirv/tessellate.vert.h"

#include "shaders/out/generated/advanced_blend.glsl.hpp"
#include "shaders/out/generated/common.glsl.hpp"
#include "shaders/out/generated/constants.glsl.hpp"
#include "shaders/out/generated/draw_image_mesh.glsl.hpp"
#include "shaders/out/generated/draw_path.glsl.hpp"
#include "shaders/out/generated/draw_path_common.glsl.hpp"
#include "shaders/out/generated/glsl.glsl.hpp"
#include "shaders/out/generated/tessellate.glsl.hpp"

#include <sstream>

namespace rive::pls {
PLSRenderContextKDGpuImpl::~PLSRenderContextKDGpuImpl() {}

// wrapper factory function MakeContext instead of constructor. just calls
// private constructor

std::unique_ptr<PLSRenderContext> PLSRenderContextKDGpuImpl::MakeContext(
    KDGpu::Device &&device, KDGpu::Queue &&queue, const ContextOptions &options,
    const pls::PlatformFeatures &baselinePlatformFeatures) {
  auto impl = std::unique_ptr<PLSRenderContextKDGpuImpl>(
      new PLSRenderContextKDGpuImpl(std::move(device), std::move(queue),
                                    options, baselinePlatformFeatures));

  return std::make_unique<PLSRenderContext>(std::move(impl));
}

// called by MakeContext
PLSRenderContextKDGpuImpl::PLSRenderContextKDGpuImpl(
    KDGpu::Device &&device, KDGpu::Queue &&queue, const ContextOptions &options,
    const pls::PlatformFeatures &baselinePlatformFeatures)
    : m_contextOptions(options), m_device(std::move(device)),
      m_queue(std::move(queue)),
      m_colorRampPipeline(std::make_unique<ColorRampPipeline>(m_device)),
      m_tessellatePipeline(
          std::make_unique<TessellatePipeline>(m_device, m_contextOptions)) {
  m_platformFeatures = baselinePlatformFeatures;
  // TODO: read baselinePlatformFeatures like supportsKHRBlendEquations

  using namespace KDGpu;

  m_imageAvailableSemaphore = m_device.createGpuSemaphore();
  m_renderCompleteSemaphore = m_device.createGpuSemaphore();
  m_frameInFlightFence =
      m_device.createFence(FenceOptions{.createSignalled = true});

  m_drawBindGroupLayouts[0] =
      m_device.createBindGroupLayout(BindGroupLayoutOptions{
          .bindings =
              {
                  {
                      .binding = TESS_VERTEX_TEXTURE_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::SampledImage,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = GRAD_TEXTURE_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::SampledImage,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::FragmentBit),
                  },
                  {
                      .binding = IMAGE_TEXTURE_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::SampledImage,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::FragmentBit),
                  },
                  {
                      .binding = PATH_BUFFER_IDX,
                      .count = 1,
                      .resourceType = m_contextOptions.disableStorageBuffers
                                          ? ResourceBindingType::SampledImage
                                          : ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = PAINT_BUFFER_IDX,
                      .count = 1,
                      .resourceType = m_contextOptions.disableStorageBuffers
                                          ? ResourceBindingType::SampledImage
                                          : ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = PAINT_AUX_BUFFER_IDX,
                      .count = 1,
                      .resourceType = m_contextOptions.disableStorageBuffers
                                          ? ResourceBindingType::SampledImage
                                          : ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = CONTOUR_BUFFER_IDX,
                      .count = 1,
                      .resourceType = m_contextOptions.disableStorageBuffers
                                          ? ResourceBindingType::SampledImage
                                          : ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = FLUSH_UNIFORM_BUFFER_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::UniformBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = IMAGE_DRAW_UNIFORM_BUFFER_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::UniformBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit) |
                          ShaderStageFlags(ShaderStageFlagBits::FragmentBit),
                  },
              },
      });

  // describe the size and number of resources for samplers in frag shader
  m_drawBindGroupLayouts[SAMPLER_BINDINGS_SET] =
      m_device.createBindGroupLayout(BindGroupLayoutOptions{
          .bindings =
              {
                  {
                      .binding = GRAD_TEXTURE_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::Sampler,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::FragmentBit),
                  },
                  {
                      .binding = IMAGE_TEXTURE_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::Sampler,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::FragmentBit),
                  },
              },
      });

  // create the samplers for each of the textures
  m_linearSampler = m_device.createSampler(SamplerOptions{
      .magFilter = FilterMode::Linear,
      .minFilter = FilterMode::Linear,
      .mipmapFilter = MipmapFilterMode::Nearest,
      .u = AddressMode::ClampToEdge,
      .v = AddressMode::ClampToEdge,
  });

  m_mipmapSampler = m_device.createSampler(SamplerOptions{
      .magFilter = FilterMode::Linear,
      .minFilter = FilterMode::Linear,
      .mipmapFilter = MipmapFilterMode::Nearest,
      .u = AddressMode::ClampToEdge,
      .v = AddressMode::ClampToEdge,
  });

  // pass in the description of size and number of samplers, and provide
  // the actual sampler handles which the bind group will keep reference of.
  m_samplerBindings = m_device.createBindGroup(BindGroupOptions{
      .layout = m_drawBindGroupLayouts[SAMPLER_BINDINGS_SET],
      .resources =
          {
              KDGpu::BindGroupEntry{
                  .binding = GRAD_TEXTURE_IDX,
                  .resource = KDGpu::SamplerBinding{.sampler = m_linearSampler},
              },
              KDGpu::BindGroupEntry{
                  .binding = IMAGE_TEXTURE_IDX,
                  .resource = KDGpu::SamplerBinding{.sampler = m_mipmapSampler},
              },
          },
  });

  // copy handles to our layouts into a vector and pass them into a pipeline
  // layout's constructor
  std::vector<Handle<BindGroupLayout_t>> layouts;
  layouts.reserve(m_drawBindGroupLayouts.size());
  for (auto &layout : m_drawBindGroupLayouts)
    layouts.emplace_back(layout);

  m_drawPipelineLayout = m_device.createPipelineLayout(PipelineLayoutOptions{
      .bindGroupLayouts = std::move(layouts),
  });

  m_tessSpanIndexBuffer = m_device.createBuffer({
      .size = sizeof(pls::kTessSpanIndices),
      .usage = BufferUsageFlagBits::IndexBufferBit,
      .memoryUsage = MemoryUsage::CpuToGpu,
  });

  {
    void *bufferData = m_tessSpanIndexBuffer.map();
    std::memcpy(bufferData, pls::kTessSpanIndices,
                sizeof(pls::kTessSpanIndices));
    m_tessSpanIndexBuffer.unmap();
  }

  m_patchVertexBuffer = m_device.createBuffer(BufferOptions{
      .size = kPatchVertexBufferCount * sizeof(PatchVertex),
      .usage = BufferUsageFlagBits::VertexBufferBit,
      .memoryUsage = MemoryUsage::CpuToGpu,
  });

  m_patchIndexBuffer = m_device.createBuffer(BufferOptions{
      .size = math::round_up_to_multiple_of<4>(kPatchIndexBufferCount *
                                               sizeof(uint16_t)),
      .usage = BufferUsageFlagBits::IndexBufferBit,
      .memoryUsage = MemoryUsage::CpuToGpu,
  });

  GeneratePatchBufferData(
      reinterpret_cast<PatchVertex *>(m_patchVertexBuffer.map()),
      reinterpret_cast<uint16_t *>(m_patchIndexBuffer.map()));

  m_patchVertexBuffer.unmap();
  m_patchIndexBuffer.unmap();

  m_nullImagePaintTexture = m_device.createTexture(TextureOptions{
      .type = TextureType::TextureType2D,
      .format = Format::R8G8B8A8_UNORM,
      .extent = {.width = 1, .height = 1, .depth = 1},
      .mipLevels = 1, // vmaCreateImage just fails if miplevels is 0
      .usage = TextureUsageFlagBits::SampledBit,
      .memoryUsage = MemoryUsage::GpuOnly,
  });

  m_nullImagePaintTextureView = m_nullImagePaintTexture.createView();
}

class RenderBufferKDGpuImpl : public RenderBuffer {
public:
  RenderBufferKDGpuImpl(KDGpu::Device &device, KDGpu::Queue &queue,
                        RenderBufferType renderBufferType,
                        RenderBufferFlags renderBufferFlags, size_t sizeInBytes)
      : RenderBuffer(renderBufferType, renderBufferFlags, sizeInBytes),
        m_device(device), m_queue(queue) {

    const bool mappedOnceAtInitialization =
        flags() & RenderBufferFlags::mappedOnceAtInitialization;
    const int bufferCount = mappedOnceAtInitialization ? 1 : m_buffers.size();
    KDGpu::BufferOptions desc = {
        // vestigal from webgpu needing multiple of four, dont think this is bad
        // practice though
        .size = math::round_up_to_multiple_of<4>(sizeInBytes),
        .usage = type() == RenderBufferType::index
                     ? KDGpu::BufferUsageFlagBits::IndexBufferBit
                     : KDGpu::BufferUsageFlagBits::VertexBufferBit,
        .memoryUsage = KDGpu::MemoryUsage::CpuToGpu,
    };
    if (!mappedOnceAtInitialization) {
      // TODO: originally CopyDst in webgpu, am guessing
      desc.usage |= KDGpu::BufferUsageFlagBits::TransferDstBit;
    }
    for (int i = 0; i < bufferCount; ++i) {
      m_buffers[i] = m_device.createBuffer(desc);
    }
  }

  const KDGpu::Buffer &submittedBuffer() const {
    return m_buffers[m_submittedBufferIdx];
  }

protected:
  void *onMap() override {
    m_submittedBufferIdx = (m_submittedBufferIdx + 1) % pls::kBufferRingSize;
    assert(m_buffers.size() > m_submittedBufferIdx);
    assert(m_buffers[m_submittedBufferIdx].isValid());
    if (flags() & RenderBufferFlags::mappedOnceAtInitialization) {
      assert(m_submittedBufferIdx == 1);
      return m_buffers[m_submittedBufferIdx].map();
    } else {
      if (m_stagingBuffer == nullptr) {
        m_stagingBuffer.reset(new uint8_t[sizeInBytes()]);
      }
      return m_stagingBuffer.get();
    }
  }

  void onUnmap() override {
    if (flags() & RenderBufferFlags::mappedOnceAtInitialization) {
      m_buffers[m_submittedBufferIdx].unmap();
    } else {
      std::memcpy(m_buffers[m_submittedBufferIdx].map(), m_stagingBuffer.get(),
                  sizeInBytes());
    }
  }

private:
  KDGpu::Device &m_device;
  KDGpu::Queue &m_queue;
  std::array<KDGpu::Buffer, pls::kBufferRingSize> m_buffers;
  int m_submittedBufferIdx = -1;
  std::unique_ptr<uint8_t[]> m_stagingBuffer;
};

rcp<RenderBuffer> PLSRenderContextKDGpuImpl::makeRenderBuffer(
    RenderBufferType type, RenderBufferFlags flags, size_t sizeInBytes) {
  return make_rcp<RenderBufferKDGpuImpl>(m_device, m_queue, type, flags,
                                         sizeInBytes);
}

// virtual wrapper around kdgpu textures
class PLSTextureKDGpuImpl : public PLSTexture {
public:
  PLSTextureKDGpuImpl(KDGpu::Device &device, KDGpu::Queue &queue,
                      uint32_t width, uint32_t height, uint32_t mipLevelCount,
                      const uint8_t imageDataRGBA[])
      : PLSTexture(width, height) {
    using namespace KDGpu;
    m_texture = device.createTexture(TextureOptions{
        .type = TextureType::TextureType2D,
        .format = Format::R8G8B8A8_UNORM,
        .extent = Extent3D{.width = width, .height = height, .depth = 1},
        .mipLevels = 1,
        // NOTE: originally
        //.usage = wgpu::TextureUsage::TextureBinding |
        // wgpu::TextureUsage::CopyDst,
        .usage = TextureUsageFlags(TextureUsageFlagBits::TransferDstBit) |
                 TextureUsageFlags(TextureUsageFlagBits::SampledBit),
        // TODO: implement mipmap generation.
        .memoryUsage = MemoryUsage::CpuToGpu,
    });

    m_textureView = m_texture.createView();

    // NOTE: not sure why 4... size is in floats? copied from webgpu
    // TODO: implement mipmap generation.
    std::memcpy(m_texture.map(), imageDataRGBA, width * height * 4);
    m_texture.unmap();
  }

  const KDGpu::TextureView &textureView() const { return m_textureView; }

private:
  KDGpu::Texture m_texture;
  KDGpu::TextureView m_textureView;
};

rcp<PLSTexture>
PLSRenderContextKDGpuImpl::makeImageTexture(uint32_t width, uint32_t height,
                                            uint32_t mipLevelCount,
                                            const uint8_t imageDataRGBA[]) {
  return make_rcp<PLSTextureKDGpuImpl>(m_device, m_queue, width, height,
                                       mipLevelCount, imageDataRGBA);
}

class BufferKDGpu : public BufferRing {
public:
  static std::unique_ptr<BufferKDGpu> Make(KDGpu::Device &device,
                                           KDGpu::Queue &queue,
                                           size_t capacityInBytes,
                                           KDGpu::BufferUsageFlags usage) {
    return std::make_unique<BufferKDGpu>(device, queue, capacityInBytes, usage);
  }

  BufferKDGpu(KDGpu::Device &device, KDGpu::Queue &queue,
              size_t capacityInBytes, KDGpu::BufferUsageFlags usage)
      : BufferRing(std::max<size_t>(capacityInBytes, 1)), m_queue(queue) {
    for (auto &buf : m_buffers) {
      buf = device.createBuffer(KDGpu::BufferOptions{
          .size = this->capacityInBytes(),
          .usage = usage | KDGpu::BufferUsageFlags(
                               // NOTE: wgpu::BufferUsage::CopyDst originally
                               KDGpu::BufferUsageFlagBits::TransferDstBit),
          .memoryUsage = KDGpu::MemoryUsage::CpuToGpu,
      });
    }
  }

  const KDGpu::Buffer &submittedBuffer() const {
    return m_buffers[submittedBufferIdx()];
  }

protected:
  void *onMapBuffer(int bufferIdx, size_t mapSizeInBytes) override {
    return shadowBuffer();
  }

  void onUnmapAndSubmitBuffer(int bufferIdx, size_t mapSizeInBytes) override {
    std::memcpy(m_buffers[bufferIdx].map(), shadowBuffer(), mapSizeInBytes);
    // NOTE: why are we not unmapping after doing this?
  }

  const KDGpu::Queue &m_queue;
  std::array<KDGpu::Buffer, kBufferRingSize> m_buffers;
};

// GL TextureFormat to use for a texture that polyfills a storage buffer.
static KDGpu::Format
storage_texture_format(pls::StorageBufferStructure bufferStructure) {
  switch (bufferStructure) {
  case pls::StorageBufferStructure::uint32x4:
    return KDGpu::Format::R32G32B32A32_UINT;
  case pls::StorageBufferStructure::uint32x2:
    return KDGpu::Format::R32G32_UINT;
  case pls::StorageBufferStructure::float32x4:
    return KDGpu::Format::R32G32B32A32_SFLOAT;
  }
  RIVE_UNREACHABLE();
}

// Buffer ring with a texture used to polyfill storage buffers when they are
// disabled.
class StorageTextureBufferKDGpu : public BufferKDGpu {
public:
  StorageTextureBufferKDGpu(KDGpu::Device &device, KDGpu::Queue &queue,
                            size_t capacityInBytes,
                            pls::StorageBufferStructure bufferStructure)
      : BufferKDGpu(
            device, queue,
            pls::StorageTextureBufferSize(capacityInBytes, bufferStructure),
            // NOTE: this originally was wgpu CopySrc
            KDGpu::BufferUsageFlagBits::TransferDstBit),
        m_bufferStructure(bufferStructure) {
    // Create a texture to mirror the buffer contents.
    auto [textureWidth, textureHeight] =
        pls::StorageTextureSize(this->capacityInBytes(), bufferStructure);

    using namespace KDGpu;
    m_texture = device.createTexture(KDGpu::TextureOptions{
        .type = TextureType::TextureType2D,
        .format = storage_texture_format(bufferStructure),
        .extent =
            {
                .width = textureWidth,
                .height = textureHeight,
                .depth = 1,
            },
        .mipLevels = 1,
        .usage = KDGpu::TextureUsageFlags(TextureUsageFlagBits::SampledBit) |
                 KDGpu::TextureUsageFlagBits::TransferDstBit,
    });
    m_textureView = m_texture.createView();
  }

  void updateTextureFromBuffer(size_t bindingSizeInBytes,
                               size_t offsetSizeInBytes,
                               KDGpu::CommandRecorder &commandRecorder) const {
    using namespace KDGpu;
    auto [updateWidth, updateHeight] =
        pls::StorageTextureSize(bindingSizeInBytes, m_bufferStructure);

    commandRecorder.copyBufferToTexture(BufferToTextureCopy{
        .srcBuffer = submittedBuffer(),
        .dstTexture = m_texture,
        .regions =
            {
                BufferTextureCopyRegion{
                    .bufferOffset = offsetSizeInBytes,
                    .bufferRowLength = (STORAGE_TEXTURE_WIDTH *
                                        pls::StorageBufferElementSizeInBytes(
                                            m_bufferStructure)),
                    .textureOffset = {0, 0, 0},
                    .textureExtent =
                        {
                            .width = updateWidth,
                            .height = updateHeight,
                            .depth = 1,
                        },
                },
            },
    });
  }

  const KDGpu::TextureView &textureView() const { return m_textureView; }

private:
  const StorageBufferStructure m_bufferStructure;
  KDGpu::Texture m_texture;
  KDGpu::TextureView m_textureView;
};

std::unique_ptr<BufferRing>
PLSRenderContextKDGpuImpl::makeUniformBufferRing(size_t capacityInBytes) {
  // Uniform blocks must be multiples of 256 bytes in size.
  // NOTE: is the above comment true only for webgpu?
  capacityInBytes = std::max<size_t>(capacityInBytes, 256);
  assert(capacityInBytes % 256 == 0);
  return std::make_unique<BufferKDGpu>(
      m_device, m_queue, capacityInBytes,
      KDGpu::BufferUsageFlagBits::UniformBufferBit);
}

std::unique_ptr<BufferRing> PLSRenderContextKDGpuImpl::makeStorageBufferRing(
    size_t capacityInBytes, pls::StorageBufferStructure bufferStructure) {
  if (m_contextOptions.disableStorageBuffers) {
    return std::make_unique<StorageTextureBufferKDGpu>(
        m_device, m_queue, capacityInBytes, bufferStructure);
  } else {
    return std::make_unique<BufferKDGpu>(
        m_device, m_queue, capacityInBytes,
        KDGpu::BufferUsageFlagBits::StorageBufferBit);
  }
}

std::unique_ptr<BufferRing>
PLSRenderContextKDGpuImpl::makeVertexBufferRing(size_t capacityInBytes) {
  return std::make_unique<BufferKDGpu>(
      m_device, m_queue, capacityInBytes,
      KDGpu::BufferUsageFlagBits::VertexBufferBit);
}

std::unique_ptr<BufferRing>
PLSRenderContextKDGpuImpl::makeTextureTransferBufferRing(
    size_t capacityInBytes) {
  return std::make_unique<BufferKDGpu>(
      m_device, m_queue, capacityInBytes,
      KDGpu::BufferUsageFlagBits::TransferSrcBit);
}

void PLSRenderContextKDGpuImpl::resizeGradientTexture(uint32_t width,
                                                      uint32_t height) {
  width = std::max(width, 1u);
  height = std::max(height, 1u);

  using namespace KDGpu;
  m_gradientTexture = m_device.createTexture(KDGpu::TextureOptions{
      .type = TextureType::TextureType2D,
      .format = Format::R8G8B8A8_UNORM,
      .extent = {.width = static_cast<uint32_t>(width),
                 .height = static_cast<uint32_t>(height),
                 .depth = 1},
      .mipLevels = 1,
      // NOTE: originally RenderAttachment and TextureBinding for wgpu
      .usage = TextureUsageFlags(TextureUsageFlagBits::ColorAttachmentBit) |
               TextureUsageFlagBits::SampledBit,
  });

  m_gradientTextureView = m_gradientTexture.createView();
}

void PLSRenderContextKDGpuImpl::resizeTessellationTexture(uint32_t width,
                                                          uint32_t height) {
  width = std::max(width, 1u);
  height = std::max(height, 1u);

  using namespace KDGpu;
  m_tesselationTexture = m_device.createTexture(KDGpu::TextureOptions{
      .type = TextureType::TextureType2D,
      .format = Format::R32G32B32A32_UINT,
      .extent = {.width = static_cast<uint32_t>(width),
                 .height = static_cast<uint32_t>(height),
                 .depth = 1},
      .mipLevels = 1,
      // NOTE: originally RenderAttachment and TextureBinding for wgpu
      .usage = TextureUsageFlags(TextureUsageFlagBits::ColorAttachmentBit) |
               TextureUsageFlagBits::SampledBit,
  });

  m_tesselationTextureView = m_tesselationTexture.createView();
}

KDGpu::RenderPassCommandRecorder PLSRenderContextKDGpuImpl::makePLSRenderPass(
    KDGpu::CommandRecorder &commandRecorder,
    const PLSRenderTargetKDGpu &renderTarget,
    KDGpu::AttachmentLoadOperation loadOp,
    const KDGpu::ColorClearValue &clearColor) {
  static_assert(FRAMEBUFFER_PLANE_IDX == 0);
  static_assert(COVERAGE_PLANE_IDX == 1);
  static_assert(CLIP_PLANE_IDX == 2);
  static_assert(ORIGINAL_DST_COLOR_PLANE_IDX == 3);
  using namespace KDGpu;

  return commandRecorder.beginRenderPass(RenderPassCommandRecorderOptions{
      .colorAttachments =
          {
              ColorAttachment{
                  // framebuffer
                  .view = renderTarget.m_targetTextureView,
                  .loadOperation = loadOp,
                  .storeOperation = AttachmentStoreOperation::Store,
                  .clearValue = clearColor,
              },
              ColorAttachment{
                  // coverage
                  .view = renderTarget.m_coverageTextureView,
                  .loadOperation = AttachmentLoadOperation::Clear,
                  .storeOperation = AttachmentStoreOperation::DontCare,
                  .clearValue = {},
              },
              ColorAttachment{
                  // clip
                  .view = renderTarget.m_clipTextureView,
                  .loadOperation = AttachmentLoadOperation::Clear,
                  .storeOperation = AttachmentStoreOperation::DontCare,
                  .clearValue = {},
              },
              ColorAttachment{
                  // originalDstColor
                  .view = renderTarget.m_originalDstColorTextureView,
                  .loadOperation = AttachmentLoadOperation::Clear,
                  .storeOperation = AttachmentStoreOperation::DontCare,
                  .clearValue = {},
              },
          },
  });
}

KDGpu::GraphicsPipeline PLSRenderContextKDGpuImpl::makePLSDrawPipeline(
    rive::pls::DrawType drawType, KDGpu::Format framebufferFormat,
    KDGpu::ShaderModule vertexShader, KDGpu::ShaderModule fragmentShader) {
  using namespace KDGpu;
  std::vector<VertexAttribute> attrs;
  std::vector<VertexBufferLayout> vertexBufferLayouts;
  switch (drawType) {
  case DrawType::midpointFanPatches:
  case DrawType::outerCurvePatches: {
    attrs = {
        {
            .location = 0,
            .format = Format::R32G32B32A32_SFLOAT,
            .offset = 0,
        },
        {
            .location = 1,
            .format = Format::R32G32B32A32_SFLOAT,
            .offset = 4 * sizeof(float),
        },
    };

    vertexBufferLayouts = {
        {
            .stride = sizeof(pls::PatchVertex),
            .inputRate = VertexRate::Vertex,
        },
    };
    break;
  }
  case DrawType::interiorTriangulation: {
    attrs = {
        {
            .location = 0,
            .format = Format::R32G32B32_SFLOAT,
            .offset = 0,
        },
    };

    vertexBufferLayouts = {
        {
            .stride = sizeof(pls::TriangleVertex),
            .inputRate = VertexRate::Vertex,
        },
    };
    break;
  }
  case DrawType::imageRect:
    RIVE_UNREACHABLE();
  case DrawType::imageMesh:
    attrs = {
        {
            .location = 0,
            .format = Format::R32G32_SFLOAT,
            .offset = 0,
        },
        {
            .location = 1,
            .format = Format::R32G32_SFLOAT,
            .offset = 0,
        },
    };

    vertexBufferLayouts = {
        {
            .stride = sizeof(float) * 2,
            .inputRate = VertexRate::Vertex,
        },
        {
            .stride = sizeof(float) * 2,
            .inputRate = VertexRate::Vertex,
        },
    };
    break;
  case DrawType::plsAtomicInitialize:
  case DrawType::plsAtomicResolve:
  case DrawType::stencilClipReset:
    RIVE_UNREACHABLE();
  }

  static_assert(FRAMEBUFFER_PLANE_IDX == 0);
  static_assert(COVERAGE_PLANE_IDX == 1);
  static_assert(CLIP_PLANE_IDX == 2);
  static_assert(ORIGINAL_DST_COLOR_PLANE_IDX == 3);

  return m_device.createGraphicsPipeline(GraphicsPipelineOptions{
      .shaderStages = {{.shaderModule = vertexShader,
                        .stage = ShaderStageFlagBits::VertexBit},
                       {.shaderModule = fragmentShader,
                        .stage = ShaderStageFlagBits::FragmentBit}},
      .layout = m_drawPipelineLayout,
      .vertex =
          {
              .buffers = vertexBufferLayouts,
              .attributes = attrs,
          },
      .renderTargets =
          {
              {.format = framebufferFormat},
              {.format = Format::R32_UINT},
              {.format = Format::R32_UINT},
              {.format = framebufferFormat},
          },
      .primitive =
          {
              .topology = PrimitiveTopology::TriangleList,
              .cullMode = drawType != DrawType::imageMesh
                              ? CullModeFlagBits::BackBit
                              : CullModeFlagBits::None,
              .frontFace = FrontFace::Clockwise,
          },
  });
}

// void PLSRenderContextKDGpuImpl::prepareToMapBuffers()

void PLSRenderTargetKDGpu::setTargetTextureView(
    KDGpu::Handle<KDGpu::TextureView_t> textureView) {
  m_targetTextureView = textureView;
}

static std::vector<uint32_t> charBufferToCode(const uint32_t *buf,
                                              size_t sizeInBytes) {
  std::vector<uint32_t> code;
  assert(sizeInBytes % sizeof(uint32_t) == 0);
  code.resize(sizeInBytes / sizeof(uint32_t));
  std::memcpy(code.data(), buf, sizeInBytes);
  return code;
}

static std::vector<uint32_t> charBufferToCode(const char *buf, size_t size) {
  std::vector<uint32_t> code;
  code.resize(std::ceil(static_cast<float>(size) / sizeof(uint32_t)));
  assert(code.size() * sizeof(uint32_t) >= size);
  std::memcpy(code.data(), buf, size);
  return code;
}

// Renders color ramps to the gradient texture.
class PLSRenderContextKDGpuImpl::ColorRampPipeline {
public:
  ColorRampPipeline(KDGpu::Device &device) {
    using namespace KDGpu;
    m_bindGroupLayout =
        device.createBindGroupLayout(KDGpu::BindGroupLayoutOptions{
            .bindings = {
                {
                    .binding = FLUSH_UNIFORM_BUFFER_IDX,
                    .count = 1,
                    .resourceType = ResourceBindingType::UniformBuffer,
                    .shaderStages =
                        ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                },
            }});

    m_pipelineLayout = device.createPipelineLayout(PipelineLayoutOptions{
        .bindGroupLayouts = {m_bindGroupLayout},
    });

    KDGpu::ShaderModule vertexShader = device.createShaderModule(
        charBufferToCode(color_ramp_vert, sizeof(color_ramp_vert)));

    KDGpu::ShaderModule fragmentShader = device.createShaderModule(
        charBufferToCode(color_ramp_frag, sizeof(color_ramp_frag)));

    m_renderPipeline = device.createGraphicsPipeline(GraphicsPipelineOptions{
        .shaderStages =
            {
                {
                    .shaderModule = vertexShader.handle(),
                    .stage = ShaderStageFlagBits::VertexBit,
                },
                {
                    .shaderModule = fragmentShader.handle(),
                    .stage = ShaderStageFlagBits::FragmentBit,
                },
            },
        .layout = m_pipelineLayout,
        .vertex =
            VertexOptions{
                .buffers = {{
                    .binding = 0,
                    .stride = sizeof(pls::GradientSpan),
                    .inputRate = VertexRate::Instance,
                }},
                .attributes =
                    {
                        {
                            .location = 0,
                            .format = Format::R32G32B32A32_UINT,
                            .offset = 0,
                        },
                    },
            },
        .renderTargets = {{.format = Format::R8G8B8A8_UNORM}},
        .primitive =
            {
                .topology = PrimitiveTopology::TriangleStrip,
                .cullMode = CullModeFlagBits::BackBit,
                .frontFace = FrontFace::Clockwise,
            },
    });
  }

  const KDGpu::BindGroupLayout &bindGroupLayout() const {
    return m_bindGroupLayout;
  }
  const KDGpu::GraphicsPipeline &renderPipeline() const {
    return m_renderPipeline;
  }

private:
  KDGpu::BindGroupLayout m_bindGroupLayout;
  // EmJsHandle m_vertexShaderHandle;
  // EmJsHandle m_fragmentShaderHandle;
  KDGpu::GraphicsPipeline m_renderPipeline;
  KDGpu::PipelineLayout m_pipelineLayout;
};

class PLSRenderContextKDGpuImpl::TessellatePipeline {
public:
  TessellatePipeline(KDGpu::Device &device,
                     const ContextOptions &contextOptions) {
    using namespace KDGpu;
    m_bindGroupLayout = device.createBindGroupLayout(BindGroupLayoutOptions{
        .bindings = {
            {
                .binding = PATH_BUFFER_IDX,
                .count = 1,
                .resourceType = contextOptions.disableStorageBuffers
                                    ? ResourceBindingType::SampledImage
                                    : ResourceBindingType::StorageBuffer,
                .shaderStages =
                    ShaderStageFlags(ShaderStageFlagBits::VertexBit),
            },
            {
                .binding = CONTOUR_BUFFER_IDX,
                .count = 1,
                .resourceType = contextOptions.disableStorageBuffers
                                    ? ResourceBindingType::SampledImage
                                    : ResourceBindingType::StorageBuffer,
                .shaderStages =
                    ShaderStageFlags(ShaderStageFlagBits::VertexBit),
            },
            {
                .binding = FLUSH_UNIFORM_BUFFER_IDX,
                .count = 1,
                .resourceType = ResourceBindingType::UniformBuffer,
                .shaderStages =
                    ShaderStageFlags(ShaderStageFlagBits::VertexBit),
            },
        }});

    ShaderModule vertexShader;
    ShaderModule fragmentShader = device.createShaderModule(
        charBufferToCode(tessellate_frag, sizeof(tessellate_frag)));

    m_pipelineLayout = device.createPipelineLayout(PipelineLayoutOptions{
        .bindGroupLayouts = {m_bindGroupLayout},
    });

    if (contextOptions.disableStorageBuffers) {
      // The built-in SPIRV does not #define DISABLE_SHADER_STORAGE_BUFFERS.
      // Recompile the tessellation shader with storage buffers disabled.
      std::ostringstream vertexGLSL;
      vertexGLSL << "#version 460\n";
      vertexGLSL << "#pragma shader_stage(vertex)\n";
      vertexGLSL << "#define " GLSL_VERTEX "\n";
      vertexGLSL << "#define " GLSL_DISABLE_SHADER_STORAGE_BUFFERS "\n";
      vertexGLSL << "#define " GLSL_TARGET_VULKAN "\n";
      vertexGLSL
          << "#extension GL_EXT_samplerless_texture_functions : enable\n";
      vertexGLSL << glsl::glsl << "\n";
      vertexGLSL << glsl::constants << "\n";
      vertexGLSL << glsl::common << "\n";
      vertexGLSL << glsl::tessellate << "\n";
      vertexShader = device.createShaderModule(charBufferToCode(
          vertexGLSL.str().c_str(), vertexGLSL.str().length()));
    } else {
      vertexShader = device.createShaderModule(
          charBufferToCode(tessellate_vert, sizeof(tessellate_vert)));
    }

    m_renderPipeline = device.createGraphicsPipeline(GraphicsPipelineOptions{
        .shaderStages =
            {
                {
                    .shaderModule = vertexShader,
                    .stage = ShaderStageFlagBits::VertexBit,
                },
                {
                    .shaderModule = fragmentShader,
                    .stage = ShaderStageFlagBits::FragmentBit,
                },
            },
        .layout = m_pipelineLayout,
        .vertex =
            VertexOptions{
                .buffers = {{
                    .binding = 0,
                    .stride = sizeof(pls::TessVertexSpan),
                    .inputRate = VertexRate::Instance,
                }},
                .attributes =
                    {
                        {
                            .location = 0,
                            .format = Format::R32G32B32A32_SFLOAT,
                            .offset = 0,
                        },
                        {
                            .location = 1,
                            .format = Format::R32G32B32A32_SFLOAT,
                            .offset = 4 * sizeof(float),
                        },
                        {
                            .location = 2,
                            .format = Format::R32G32B32A32_SFLOAT,
                            .offset = 8 * sizeof(float),
                        },
                        {
                            .location = 3,
                            .format = Format::R32G32B32A32_UINT,
                            .offset = 12 * sizeof(float),
                        },
                    },
            },
        .renderTargets = {{.format = Format::R32G32B32A32_UINT}},
        .primitive =
            {
                .topology = PrimitiveTopology::TriangleStrip,
                .cullMode = CullModeFlagBits::BackBit,
                .frontFace = FrontFace::Clockwise,
            },
    });
  };

  const KDGpu::BindGroupLayout &bindGroupLayout() const {
    return m_bindGroupLayout;
  }
  const KDGpu::GraphicsPipeline &renderPipeline() const {
    return m_renderPipeline;
  }

private:
  KDGpu::BindGroupLayout m_bindGroupLayout;
  KDGpu::GraphicsPipeline m_renderPipeline;
  KDGpu::PipelineLayout m_pipelineLayout;
};

class PLSRenderContextKDGpuImpl::DrawPipeline {
public:
  DrawPipeline(PLSRenderContextKDGpuImpl &context, DrawType drawType,
               pls::ShaderFeatures shaderFeatures,
               const ContextOptions &contextOptions) {
    PixelLocalStorageType plsType = context.m_contextOptions.plsType;
    KDGpu::ShaderModule vertexShader, fragmentShader;

    if (contextOptions.disableStorageBuffers) {

      const char *language = "glsl";
      const char *versionString = "#version 460";

      std::ostringstream glsl;
      auto addDefine = [&glsl](const char *name) {
        glsl << "#define " << name << "\n";
      };
      glsl << "#extension GL_EXT_samplerless_texture_functions : enable\n";
      addDefine(GLSL_TARGET_VULKAN);
      addDefine(GLSL_PLS_IMPL_NONE);

      if (contextOptions.disableStorageBuffers) {
        addDefine(GLSL_DISABLE_SHADER_STORAGE_BUFFERS);
      }
      switch (drawType) {
      case DrawType::midpointFanPatches:
      case DrawType::outerCurvePatches:
        addDefine(GLSL_ENABLE_INSTANCE_INDEX);
        break;
      case DrawType::interiorTriangulation:
        addDefine(GLSL_DRAW_INTERIOR_TRIANGLES);
        break;
      case DrawType::imageRect:
        RIVE_UNREACHABLE();
      case DrawType::imageMesh:
        break;
      case DrawType::plsAtomicInitialize:
      case DrawType::plsAtomicResolve:
      case DrawType::stencilClipReset:
        RIVE_UNREACHABLE();
      }
      for (size_t i = 0; i < pls::kShaderFeatureCount; ++i) {
        ShaderFeatures feature = static_cast<ShaderFeatures>(1 << i);
        if (shaderFeatures & feature) {
          addDefine(GetShaderFeatureGLSLName(feature));
        }
      }
      glsl << pls::glsl::glsl << '\n';
      glsl << pls::glsl::constants << '\n';
      glsl << pls::glsl::common << '\n';
      if (shaderFeatures & ShaderFeatures::ENABLE_ADVANCED_BLEND) {
        glsl << pls::glsl::advanced_blend << '\n';
      }
      if (context.platformFeatures().avoidFlatVaryings) {
        addDefine(GLSL_OPTIONALLY_FLAT);
      } else {
        glsl << "#define " GLSL_OPTIONALLY_FLAT " flat\n";
      }
      switch (drawType) {
      case DrawType::midpointFanPatches:
      case DrawType::outerCurvePatches:
        addDefine(GLSL_DRAW_PATH);
        glsl << pls::glsl::draw_path_common << '\n';
        glsl << pls::glsl::draw_path << '\n';
        break;
      case DrawType::interiorTriangulation:
        addDefine(GLSL_DRAW_INTERIOR_TRIANGLES);
        glsl << pls::glsl::draw_path_common << '\n';
        glsl << pls::glsl::draw_path << '\n';
        break;
      case DrawType::imageRect:
        addDefine(GLSL_DRAW_IMAGE);
        addDefine(GLSL_DRAW_IMAGE_RECT);
        RIVE_UNREACHABLE();
      case DrawType::imageMesh:
        addDefine(GLSL_DRAW_IMAGE);
        addDefine(GLSL_DRAW_IMAGE_MESH);
        glsl << pls::glsl::draw_image_mesh << '\n';
        break;
      case DrawType::plsAtomicInitialize:
        addDefine(GLSL_DRAW_RENDER_TARGET_UPDATE_BOUNDS);
        addDefine(GLSL_INITIALIZE_PLS);
        RIVE_UNREACHABLE();
      case DrawType::plsAtomicResolve:
        addDefine(GLSL_DRAW_RENDER_TARGET_UPDATE_BOUNDS);
        addDefine(GLSL_RESOLVE_PLS);
        RIVE_UNREACHABLE();
      case DrawType::stencilClipReset:
        RIVE_UNREACHABLE();
      }

      std::ostringstream vertexGLSL;
      vertexGLSL << versionString << "\n";
      vertexGLSL << "#pragma shader_stage(vertex)\n";
      vertexGLSL << "#define " GLSL_VERTEX "\n";
      vertexGLSL << glsl.str();
      vertexShader = context.device().createShaderModule(
          charBufferToCode(vertexGLSL.str().c_str(), vertexGLSL.str().size()));

      std::ostringstream fragmentGLSL;
      fragmentGLSL << versionString << "\n";
      fragmentGLSL << "#pragma shader_stage(fragment)\n";
      fragmentGLSL << "#define " GLSL_FRAGMENT "\n";
      fragmentGLSL << glsl.str();
      fragmentShader = context.device().createShaderModule(charBufferToCode(
          fragmentGLSL.str().c_str(), fragmentGLSL.str().size()));
    } else {
      switch (drawType) {
      case DrawType::midpointFanPatches:
      case DrawType::outerCurvePatches:
        vertexShader = context.device().createShaderModule(
            charBufferToCode(draw_path_vert, sizeof(draw_path_vert)));
        fragmentShader = context.device().createShaderModule(
            charBufferToCode(draw_path_frag, sizeof(draw_path_frag)));
        break;
      case DrawType::interiorTriangulation:
        vertexShader = context.device().createShaderModule(
            charBufferToCode(draw_interior_triangles_vert,
                             sizeof(draw_interior_triangles_vert)));
        fragmentShader = context.device().createShaderModule(
            charBufferToCode(draw_interior_triangles_frag,
                             sizeof(draw_interior_triangles_frag)));
        break;
      case DrawType::imageRect:
        RIVE_UNREACHABLE();
      case DrawType::imageMesh:
        vertexShader = context.device().createShaderModule(charBufferToCode(
            draw_image_mesh_vert, sizeof(draw_image_mesh_vert)));
        fragmentShader = context.device().createShaderModule(charBufferToCode(
            draw_image_mesh_frag, sizeof(draw_image_mesh_frag)));
        break;
      case DrawType::plsAtomicInitialize:
      case DrawType::plsAtomicResolve:
      case DrawType::stencilClipReset:
        RIVE_UNREACHABLE();
      }
    }

    for (auto framebufferFormat :
         {KDGpu::Format::B8G8R8A8_UNORM, KDGpu::Format::R8G8B8A8_UNORM}) {
      int pipelineIdx = RenderPipelineIdx(framebufferFormat);
      m_renderPipelines[pipelineIdx] = context.makePLSDrawPipeline(
          drawType, framebufferFormat, std::move(vertexShader),
          std::move(fragmentShader));
    }
  }

  const KDGpu::GraphicsPipeline &
  renderPipeline(KDGpu::Format framebufferFormat) const {
    return m_renderPipelines[RenderPipelineIdx(framebufferFormat)];
  }

private:
  static int RenderPipelineIdx(KDGpu::Format framebufferFormat) {
    assert(framebufferFormat == KDGpu::Format::B8G8R8A8_UNORM ||
           framebufferFormat == KDGpu::Format::R8G8B8A8_UNORM);
    return framebufferFormat == KDGpu::Format::B8G8R8A8_UNORM ? 1 : 0;
  }

  std::array<KDGpu::GraphicsPipeline, 2> m_renderPipelines;
};

PLSRenderTargetKDGpu::PLSRenderTargetKDGpu(
    KDGpu::Device &device, KDGpu::Format framebufferFormat, uint32_t width,
    uint32_t height, KDGpu::TextureUsageFlags additionalTextureFlags)
    : PLSRenderTarget(width, height), m_framebufferFormat(framebufferFormat) {
  using namespace KDGpu;
  KDGpu::TextureOptions desc{
      .type = TextureType::TextureType2D,
      .format = Format::R32_UINT,
      .extent = {.width = static_cast<uint32_t>(width),
                 .height = static_cast<uint32_t>(height),
                 .depth = 1},
      .mipLevels = 1,
      .usage = TextureUsageFlags(TextureUsageFlagBits::ColorAttachmentBit) |
               additionalTextureFlags,
  };

  m_coverageTexture = device.createTexture(desc);
  m_clipTexture = device.createTexture(desc);

  desc.format = m_framebufferFormat;
  m_originalDstColorTexture = device.createTexture(desc);

  m_targetTextureView = {};
  m_coverageTextureView = m_coverageTexture.createView();
  m_clipTextureView = m_clipTexture.createView();
  m_originalDstColorTextureView = m_originalDstColorTexture.createView();
}

rcp<PLSRenderTargetKDGpu>
PLSRenderContextKDGpuImpl::makeRenderTarget(KDGpu::Format framebufferFormat,
                                            uint32_t width, uint32_t height) {
  return rcp(
      new PLSRenderTargetKDGpu(m_device, framebufferFormat, width, height, {}));
}

static const KDGpu::Buffer &vulkan_buffer(const BufferRing *bufferRing) {
  assert(bufferRing != nullptr);
  return static_cast<const BufferKDGpu *>(bufferRing)->submittedBuffer();
}

template <typename HighLevelStruct>
void update_storage_texture(const BufferRing *bufferRing, size_t bindingCount,
                            size_t firstElement,
                            KDGpu::CommandRecorder &recorder) {
  assert(bufferRing != nullptr);
  auto storageTextureBuffer =
      static_cast<const StorageTextureBufferKDGpu *>(bufferRing);
  storageTextureBuffer->updateTextureFromBuffer(
      bindingCount * sizeof(HighLevelStruct),
      firstElement * sizeof(HighLevelStruct), recorder);
}

void PLSRenderContextKDGpuImpl::flush(const FlushDescriptor &desc) {
  using namespace KDGpu;
  auto *renderTarget =
      static_cast<const PLSRenderTargetKDGpu *>(desc.renderTarget);
  CommandRecorder commandRecorder = m_device.createCommandRecorder();

  // If storage buffers are disabled, copy their contents to textures.
  if (m_contextOptions.disableStorageBuffers) {
    if (desc.pathCount > 0) {
      update_storage_texture<pls::PathData>(pathBufferRing(), desc.pathCount,
                                            desc.firstPath, commandRecorder);
      update_storage_texture<pls::PaintData>(paintBufferRing(), desc.pathCount,
                                             desc.firstPaint, commandRecorder);
      update_storage_texture<pls::PaintAuxData>(
          paintAuxBufferRing(), desc.pathCount, desc.firstPaintAux,
          commandRecorder);
    }
    if (desc.contourCount > 0) {
      update_storage_texture<pls::ContourData>(
          contourBufferRing(), desc.contourCount, desc.firstContour,
          commandRecorder);
    }
  }
  // bindgroup for gradient pass which must live after the if statement
  // enclosing the gradpass
  BindGroup gradientBindings;

  // Render the complex color ramps to the gradient texture.
  if (desc.complexGradSpanCount > 0) {
    const auto &uniformBuffers =
        *static_cast<const BufferKDGpu *>(flushUniformBufferRing());

    const auto &gradSpanBuffers =
        *static_cast<const BufferKDGpu *>(gradSpanBufferRing());

    gradientBindings = m_device.createBindGroup(BindGroupOptions{
        .layout = m_colorRampPipeline->bindGroupLayout(),
        .resources =
            {
                BindGroupEntry{
                    .binding = FLUSH_UNIFORM_BUFFER_IDX,
                    .resource =
                        UniformBufferBinding{
                            .buffer = uniformBuffers.submittedBuffer(),
                            .offset = static_cast<uint32_t>(
                                desc.flushUniformDataOffsetInBytes),
                        },
                },
            },
    });

    KDGpu::RenderPassCommandRecorder gradPass =
        commandRecorder.beginRenderPass(RenderPassCommandRecorderOptions{
            .colorAttachments = {ColorAttachment{
                .view = m_gradientTextureView,
                .loadOperation = AttachmentLoadOperation::Clear,
                .storeOperation = AttachmentStoreOperation::Store,
                .clearValue = ColorClearValue{},
            }}});

    gradPass.setViewport(Viewport{
        .x = 0.f,
        .y = static_cast<float>(desc.complexGradRowsTop),
        .width = static_cast<float>(pls::kGradTextureWidth),
        .height = static_cast<float>(desc.complexGradRowsTop),
    });

    gradPass.setPipeline(m_colorRampPipeline->renderPipeline());
    gradPass.setVertexBuffer(0, gradSpanBuffers.submittedBuffer());
    gradPass.setBindGroup(0, gradientBindings);
    gradPass.draw(DrawCommand{
        .vertexCount = 4,
        .instanceCount = static_cast<uint32_t>(desc.complexGradSpanCount),
        .firstVertex = 0,
        .firstInstance = static_cast<uint32_t>(desc.firstComplexGradSpan),
    });
    gradPass.end();
  }

  // copy the simple color ramps to the gradient texture
  if (desc.simpleGradTexelsHeight > 0) {
    const auto &simpleColorRampsBuffers =
        *static_cast<const BufferKDGpu *>(simpleColorRampsBufferRing());

    commandRecorder.copyBufferToTexture(BufferToTextureCopy{
        .srcBuffer = simpleColorRampsBuffers.submittedBuffer(),
        .dstTexture = m_gradientTexture,
        .regions = {
            BufferTextureCopyRegion{
                .bufferOffset = desc.simpleGradDataOffsetInBytes,
                .textureExtent =
                    {
                        .width = desc.simpleGradTexelsWidth,
                        .height = desc.simpleGradTexelsHeight,
                        .depth = 1,
                    }},
        }});
  }

  // bindgroup used for tesselation which must live past the enclosing if
  // statement
  KDGpu::BindGroup tesselationBindings;

  // Tessellate all curves into vertices in the tessellation texture.
  if (desc.tessVertexSpanCount > 0) {
    const auto &pathBuffers =
        *static_cast<const BufferKDGpu *>(pathBufferRing());

    auto getPathBufferStorageTextureView = [&]() -> const TextureView & {
      assert(m_contextOptions.disableStorageBuffers);
      return static_cast<const StorageTextureBufferKDGpu *>(
                 std::addressof(pathBuffers))
          ->textureView();
    };

    auto getPathBufferBinding = [&]() -> BindingResource {
      if (m_contextOptions.disableStorageBuffers) {
        return TextureViewBinding{getPathBufferStorageTextureView()};
      } else {
        return StorageBufferBinding{
            .buffer = pathBuffers.submittedBuffer(),
            .offset =
                static_cast<uint32_t>(desc.firstPath * sizeof(pls::PathData)),
        };
      }
    };

    const auto &contourBuffers =
        *static_cast<const BufferKDGpu *>(contourBufferRing());

    auto getContourBufferStorageTextureView = [&]() -> const TextureView & {
      assert(m_contextOptions.disableStorageBuffers);
      return static_cast<const StorageTextureBufferKDGpu *>(
                 std::addressof(contourBuffers))
          ->textureView();
    };

    auto getContourBufferBinding = [&]() -> BindingResource {
      if (m_contextOptions.disableStorageBuffers) {
        return TextureViewBinding{getContourBufferStorageTextureView()};
      } else {
        return StorageBufferBinding{
            .buffer = pathBuffers.submittedBuffer(),
            .offset =
                static_cast<uint32_t>(desc.firstPath * sizeof(pls::PathData)),
        };
      }
    };

    tesselationBindings = m_device.createBindGroup(BindGroupOptions{
        .layout = m_tessellatePipeline->bindGroupLayout(),
        .resources =
            {
                BindGroupEntry{
                    .binding = PATH_BUFFER_IDX,
                    .resource = getPathBufferBinding(),
                },
                BindGroupEntry{
                    .binding = CONTOUR_BUFFER_IDX,
                    .resource = getContourBufferBinding(),
                },
                BindGroupEntry{
                    .binding = FLUSH_UNIFORM_BUFFER_IDX,
                    .resource =
                        UniformBufferBinding{
                            .buffer = static_cast<const BufferKDGpu *>(
                                          flushUniformBufferRing())
                                          ->submittedBuffer(),
                            .offset = static_cast<uint32_t>(
                                desc.flushUniformDataOffsetInBytes),
                        },
                },
            },
    });

    RenderPassCommandRecorder tessPass =
        commandRecorder.beginRenderPass(RenderPassCommandRecorderOptions{
            .colorAttachments =
                {
                    ColorAttachment{
                        .view = m_tesselationTextureView,
                        .loadOperation = AttachmentLoadOperation::Clear,
                        .storeOperation = AttachmentStoreOperation::Store,
                        .clearValue = ColorClearValue{},
                    },
                },
        });

    tessPass.setViewport(Viewport{
        .x = 0.f,
        .y = 0.f,
        .width = pls::kTessTextureWidth,
        .height = static_cast<float>(desc.tessDataHeight),
    });
    tessPass.setPipeline(m_tessellatePipeline->renderPipeline());
    tessPass.setVertexBuffer(
        0, static_cast<const BufferKDGpu *>(tessSpanBufferRing())
               ->submittedBuffer());
    tessPass.setIndexBuffer(m_tessSpanIndexBuffer, 0, IndexType::Uint16);
    tessPass.setBindGroup(0, tesselationBindings);
    tessPass.drawIndexed(DrawIndexedCommand{
        .indexCount = std::size(pls::kTessSpanIndices),
        .instanceCount = static_cast<uint32_t>(desc.tessVertexSpanCount),
        .firstIndex = 0,
        .vertexOffset = 0,
        .firstInstance = static_cast<uint32_t>(desc.firstTessVertexSpan),
    });
    tessPass.end();
  }

  AttachmentLoadOperation loadOp;
  ColorClearValue clearColor;
  if (desc.colorLoadAction == LoadAction::clear) {
    loadOp = AttachmentLoadOperation::Clear;
    float cc[4];
    UnpackColorToRGBA32F(desc.clearColor, cc);
    clearColor = ColorClearValue{.float32 = {cc[0], cc[1], cc[2], cc[3]}};
  } else {
    loadOp = AttachmentLoadOperation::Load;
  }

  RenderPassCommandRecorder drawPass =
      makePLSRenderPass(commandRecorder, *renderTarget, loadOp, clearColor);

  drawPass.setViewport(Viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<float>(renderTarget->width()),
      .height = static_cast<float>(renderTarget->height()),
  });

  const TextureView *currentImageTextureView = &m_nullImagePaintTextureView;
  // bindings for the draw pass which must live after everything else
  BindGroup bindings;
  bool needsNewBindings = true;

  for (const DrawBatch &batch : *desc.drawList) {
    if (batch.elementCount == 0) {
      continue;
    }

    DrawType drawType = batch.drawType;

    const DrawPipeline &drawPipeline =
        m_drawPipelines
            .try_emplace(
                pls::ShaderUniqueKey(drawType, batch.shaderFeatures,
                                     pls::InterlockMode::rasterOrdering,
                                     pls::ShaderMiscFlags::none),
                *this, drawType, batch.shaderFeatures, m_contextOptions)
            .first->second;
    drawPass.setPipeline(
        drawPipeline.renderPipeline(renderTarget->framebufferFormat()));

    // set sampler bindings per-batch
    drawPass.setBindGroup(SAMPLER_BINDINGS_SET, m_samplerBindings);

    // Bind the appropriate image texture, if any.
    if (auto *imageTexture =
            static_cast<const PLSTextureKDGpuImpl *>(batch.imageTexture)) {
      currentImageTextureView = &imageTexture->textureView();
      needsNewBindings = true;
    }

    if (needsNewBindings) {
      bindings = m_device.createBindGroup(BindGroupOptions{
               .layout = m_drawBindGroupLayouts[0],
               .resources = {
                BindGroupEntry{
                    .binding = TESS_VERTEX_TEXTURE_IDX,
                    .resource = TextureViewBinding{m_tesselationTextureView},
                },
                BindGroupEntry{
                    .binding = GRAD_TEXTURE_IDX,
                    .resource = TextureViewBinding{m_gradientTextureView},
                },
                BindGroupEntry{
                    .binding = IMAGE_TEXTURE_IDX,
                    .resource = TextureViewBinding{*currentImageTextureView},
                },
                m_contextOptions.disableStorageBuffers ?
                    KDGpu::BindGroupEntry{
                        .binding = PATH_BUFFER_IDX,
                        .resource = TextureViewBinding{static_cast<const StorageTextureBufferKDGpu*>(pathBufferRing())->textureView() },
                    } :
                    KDGpu::BindGroupEntry{
                        .binding = PATH_BUFFER_IDX,
                        .resource = StorageBufferBinding{
                            .buffer = static_cast<const BufferKDGpu*>(pathBufferRing())->submittedBuffer(),
                            .offset = static_cast<uint32_t>(desc.firstPath * sizeof(pls::PathData)),
                        },
                    },
                m_contextOptions.disableStorageBuffers ?
                    KDGpu::BindGroupEntry{
                        .binding = PAINT_BUFFER_IDX,
                        .resource = TextureViewBinding{static_cast<const StorageTextureBufferKDGpu*>(paintBufferRing())->textureView() },
                    } :
                    KDGpu::BindGroupEntry{
                        .binding = PAINT_BUFFER_IDX,
                        .resource = StorageBufferBinding{
                            .buffer = static_cast<const BufferKDGpu*>(paintBufferRing())->submittedBuffer(),
                            .offset = static_cast<uint32_t>(desc.firstPaint * sizeof(pls::PaintData)),
                        },
                    },
                m_contextOptions.disableStorageBuffers ?
                    KDGpu::BindGroupEntry{
                        .binding = PAINT_AUX_BUFFER_IDX,
                        .resource = TextureViewBinding{static_cast<const StorageTextureBufferKDGpu*>(paintAuxBufferRing())->textureView() },
                    } :
                    KDGpu::BindGroupEntry{
                        .binding = PAINT_AUX_BUFFER_IDX,
                        .resource = StorageBufferBinding{
                            .buffer = static_cast<const BufferKDGpu*>(paintAuxBufferRing())->submittedBuffer(),
                            .offset = static_cast<uint32_t>(desc.firstPaintAux * sizeof(pls::PaintAuxData)),
                        },
                    },
                m_contextOptions.disableStorageBuffers ?
                    KDGpu::BindGroupEntry{
                        .binding = CONTOUR_BUFFER_IDX,
                        .resource = TextureViewBinding{static_cast<const StorageTextureBufferKDGpu*>(contourBufferRing())->textureView() },
                    } :
                    KDGpu::BindGroupEntry{
                        .binding = CONTOUR_BUFFER_IDX,
                        .resource = StorageBufferBinding{
                            .buffer = static_cast<const BufferKDGpu*>(contourBufferRing())->submittedBuffer(),
                            .offset = static_cast<uint32_t>(desc.firstContour * sizeof(pls::ContourData)),
                        },
                    },
                {
                    .binding = FLUSH_UNIFORM_BUFFER_IDX,
                    .resource = UniformBufferBinding{
                        .buffer = static_cast<const BufferKDGpu*>(flushUniformBufferRing())->submittedBuffer(),
                        .offset = static_cast<uint32_t>(desc.flushUniformDataOffsetInBytes),
                    },
                },
                {
                    .binding = IMAGE_DRAW_UNIFORM_BUFFER_IDX,
                    .resource = UniformBufferBinding{
                        .buffer = static_cast<const BufferKDGpu*>(imageDrawUniformBufferRing())->submittedBuffer(),
                        .size = sizeof(pls::ImageDrawUniforms),
                    },
                },
        }});

      if (needsNewBindings || drawType == DrawType::imageRect ||
          drawType == DrawType::imageMesh) {
        drawPass.setBindGroup(0, bindings, {}, {batch.imageDrawDataOffset});
        needsNewBindings = false;
      }

      switch (drawType) {
      case DrawType::midpointFanPatches:
      case DrawType::outerCurvePatches: {
        // Draw PLS patches that connect the tessellation vertices.
        drawPass.setVertexBuffer(0, m_patchVertexBuffer);
        drawPass.setIndexBuffer(m_patchIndexBuffer, 0, IndexType::Uint16);
        drawPass.drawIndexed(DrawIndexedCommand{
            .indexCount = pls::PatchIndexCount(drawType),
            .instanceCount = batch.elementCount,
            .firstIndex = static_cast<uint32_t>(pls::PatchBaseIndex(drawType)),
            // NOTE: is vertexOffset here the same as baseVertex in webgpu?
            .vertexOffset = 0,
            .firstInstance = batch.baseElement,
        });
        break;
      case DrawType::interiorTriangulation: {
        drawPass.setVertexBuffer(
            0, static_cast<const BufferKDGpu *>(triangleBufferRing())
                   ->submittedBuffer());
        drawPass.draw(DrawCommand{
            .vertexCount = batch.elementCount,
            .instanceCount = 1,
            .firstVertex = batch.baseElement,
        });
        break;
      }
      case DrawType::imageRect:
        RIVE_UNREACHABLE();
      case DrawType::imageMesh: {
        auto vertexBuffer =
            static_cast<const RenderBufferKDGpuImpl *>(batch.vertexBuffer);
        auto uvBuffer =
            static_cast<const RenderBufferKDGpuImpl *>(batch.uvBuffer);
        auto indexBuffer =
            static_cast<const RenderBufferKDGpuImpl *>(batch.indexBuffer);
        drawPass.setVertexBuffer(0, vertexBuffer->submittedBuffer());
        drawPass.setVertexBuffer(1, uvBuffer->submittedBuffer());
        drawPass.setIndexBuffer(indexBuffer->submittedBuffer(), 0,
                                IndexType::Uint16);
        drawPass.drawIndexed(DrawIndexedCommand{
            .indexCount = batch.elementCount,
            .instanceCount = 1,
            .firstIndex = batch.baseElement,
        });
        break;
      }
      case DrawType::plsAtomicInitialize:
      case DrawType::plsAtomicResolve:
      case DrawType::stencilClipReset:
        RIVE_UNREACHABLE();
      }
      }
    }
  }

  drawPass.end();
  m_frameInFlightFence.reset();
  m_commandBuffer = commandRecorder.finish();
  m_queue.submit(SubmitOptions{
      .commandBuffers = {m_commandBuffer},
      .waitSemaphores = {m_imageAvailableSemaphore},
      .signalSemaphores = {m_renderCompleteSemaphore},
      .signalFence = m_frameInFlightFence,
  });
}
} // namespace rive::pls
