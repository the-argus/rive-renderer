#include "rive/renderer/kdgpu/render_context_kdgpu_impl.hpp"
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
#include <shaderc/shaderc.hpp>

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

// glsl for subpassLoad, where shaders are generated on the fly
#include "shaders/constants.glsl"
#include "shaders/out/generated/advanced_blend.glsl.hpp"
#include "shaders/out/generated/common.glsl.hpp"
#include "shaders/out/generated/constants.glsl.hpp"
#include "shaders/out/generated/draw_image_mesh.glsl.hpp"
#include "shaders/out/generated/draw_path.glsl.hpp"
#include "shaders/out/generated/draw_path_common.glsl.hpp"
#include "shaders/out/generated/glsl.glsl.hpp"

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
          .label = "drawbindGroupLayouts[0]",
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
                      .resourceType = ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = PAINT_BUFFER_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = PAINT_AUX_BUFFER_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::StorageBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit),
                  },
                  {
                      .binding = CONTOUR_BUFFER_IDX,
                      .count = 1,
                      .resourceType = ResourceBindingType::StorageBuffer,
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
                      .resourceType = ResourceBindingType::DynamicUniformBuffer,
                      .shaderStages =
                          ShaderStageFlags(ShaderStageFlagBits::VertexBit) |
                          ShaderStageFlags(ShaderStageFlagBits::FragmentBit),
                  },
              },
      });

  if (m_contextOptions.plsType == PixelLocalStorageType::subpassLoad) {
    static_assert(FRAMEBUFFER_PLANE_IDX == 0);
    static_assert(COVERAGE_PLANE_IDX == 1);
    static_assert(CLIP_PLANE_IDX == 2);
    static_assert(ORIGINAL_DST_COLOR_PLANE_IDX == 3);
    m_drawBindGroupLayouts[PLS_TEXTURE_BINDINGS_SET] =
        m_device.createBindGroupLayout(BindGroupLayoutOptions{
            .label = "drawbindGroupLayouts[PLS_TEXTURE_BINDINGS_SET]",
            .bindings =
                {
                    ResourceBindingLayout{
                        .binding = FRAMEBUFFER_PLANE_IDX,
                        .count = 1,
                        .resourceType = ResourceBindingType::InputAttachment,
                        .shaderStages = ShaderStageFlagBits::FragmentBit,
                    },
                    ResourceBindingLayout{
                        .binding = COVERAGE_PLANE_IDX,
                        .count = 1,
                        .resourceType = ResourceBindingType::InputAttachment,
                        .shaderStages = ShaderStageFlagBits::FragmentBit,
                    },
                    ResourceBindingLayout{
                        .binding = CLIP_PLANE_IDX,
                        .count = 1,
                        .resourceType = ResourceBindingType::SampledImage,
                        .shaderStages = ShaderStageFlagBits::FragmentBit,
                    },
                    ResourceBindingLayout{
                        .binding = ORIGINAL_DST_COLOR_PLANE_IDX,
                        .count = 1,
                        .resourceType = ResourceBindingType::InputAttachment,
                        .shaderStages = ShaderStageFlagBits::FragmentBit,
                    },
                },
        });
  };

  // describe the size and number of resources for samplers in frag shader
  m_drawBindGroupLayouts[SAMPLER_BINDINGS_SET] =
      m_device.createBindGroupLayout(BindGroupLayoutOptions{
          .label = "drawbindGroupLayouts[SAMPLER_BINDINGS_SET]",
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
      .label = "m_linearSampler",
      .magFilter = FilterMode::Linear,
      .minFilter = FilterMode::Linear,
      .mipmapFilter = MipmapFilterMode::Nearest,
      .u = AddressMode::ClampToEdge,
      .v = AddressMode::ClampToEdge,
  });

  m_mipmapSampler = m_device.createSampler(SamplerOptions{
      .label = "mipmapSampler",
      .magFilter = FilterMode::Linear,
      .minFilter = FilterMode::Linear,
      .mipmapFilter = MipmapFilterMode::Nearest,
      .u = AddressMode::ClampToEdge,
      .v = AddressMode::ClampToEdge,
  });

  // pass in the description of size and number of samplers, and provide
  // the actual sampler handles which the bind group will keep reference of.
  m_samplerBindings = m_device.createBindGroup(BindGroupOptions{
      .label = "samplerBindings",
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
  // layout's constructor. skip pls bindings if they are unused
  std::vector<Handle<BindGroupLayout_t>> layouts;
  layouts.reserve(m_drawBindGroupLayouts.size());
  assert(m_drawBindGroupLayouts.size() >= 2);
  layouts.emplace_back(m_drawBindGroupLayouts[0]);
  layouts.emplace_back(m_drawBindGroupLayouts[1]);
  if (m_contextOptions.plsType == PixelLocalStorageType::subpassLoad) {
    assert(m_drawBindGroupLayouts.size() >= 3);
    layouts.emplace_back(m_drawBindGroupLayouts[2]);
  }

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
      .label = "nullImagePaintTexture",
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
      desc.usage |= KDGpu::BufferUsageFlagBits::TransferSrcBit;
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
        .label = "UNKNOWN_TEXTURE",
        .type = TextureType::TextureType2D,
        .format = Format::R8G8B8A8_UNORM,
        .extent = Extent3D{.width = width, .height = height, .depth = 1},
        // TODO: implement mipmap generation.
        .mipLevels = 1,
        // NOTE: originally
        //.usage = wgpu::TextureUsage::TextureBinding |
        // wgpu::TextureUsage::CopyDst,
        .usage = TextureUsageFlags(TextureUsageFlagBits::TransferSrcBit) |
                 TextureUsageFlags(TextureUsageFlagBits::SampledBit),
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
                               KDGpu::BufferUsageFlagBits::TransferSrcBit),
          .memoryUsage = KDGpu::MemoryUsage::CpuToGpu,
      });
    }
  }

  const KDGpu::Buffer &submittedBuffer() const {
    return m_buffers[submittedBufferIdx()];
  }

  ~BufferKDGpu() {
    for (auto &buf : m_buffers) {
      buf.unmap();
    }
  }

protected:
  void *onMapBuffer(int bufferIdx, size_t mapSizeInBytes) override {
    return shadowBuffer();
  }

  void onUnmapAndSubmitBuffer(int bufferIdx, size_t mapSizeInBytes) override {
    std::memcpy(m_buffers[bufferIdx].map(), shadowBuffer(), mapSizeInBytes);
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
  return std::make_unique<BufferKDGpu>(
      m_device, m_queue, capacityInBytes,
      KDGpu::BufferUsageFlagBits::StorageBufferBit);
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
      .label = "gradientTexture",
      .type = TextureType::TextureType2D,
      .format = Format::R8G8B8A8_UNORM,
      .extent = {.width = static_cast<uint32_t>(width),
                 .height = static_cast<uint32_t>(height),
                 .depth = 1},
      .mipLevels = 1,
      // NOTE: originally RenderAttachment and TextureBinding for wgpu
      .usage = TextureUsageFlags(TextureUsageFlagBits::ColorAttachmentBit) |
               TextureUsageFlagBits::SampledBit |
               TextureUsageFlagBits::TransferDstBit,
  });

  m_gradientTextureView = m_gradientTexture.createView();
}

void PLSRenderContextKDGpuImpl::resizeTessellationTexture(uint32_t width,
                                                          uint32_t height) {
  width = std::max(width, 1u);
  height = std::max(height, 1u);

  using namespace KDGpu;
  m_tesselationTexture = m_device.createTexture(KDGpu::TextureOptions{
      .label = "tesselationTexture",
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
                  .finalLayout = TextureLayout::PresentSrc,
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
    const KDGpu::ShaderModule &vertexShader,
    const KDGpu::ShaderModule &fragmentShader) {
  using namespace KDGpu;
  std::vector<VertexAttribute> attrs;
  std::vector<VertexBufferLayout> vertexBufferLayouts;
  bool sbpLoad = m_contextOptions.plsType == PixelLocalStorageType::subpassLoad;
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
            // .stride = sizeof(pls::PatchVertex),
            // .stride = 8 * sizeof(float),
            .stride = static_cast<uint32_t>(sbpLoad ? 8 * sizeof(float)
                                                    : sizeof(pls::PatchVertex)),
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
  case DrawType::imageRect: // 3
    if (sbpLoad) {
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
          // NOTE: in the webgpu version of this, they assign this first vertex
          // buffer layout to only be associated with the first of the two
          // vertex attributes written above, and the second buffer layout only
          // with the second attribute.
          {
              .stride = 2 * sizeof(float),
              .inputRate = VertexRate::Vertex,
          },
          {
              .stride = 2 * sizeof(float),
              .inputRate = VertexRate::Vertex,
          },
      };
    } else {
      RIVE_UNREACHABLE();
    }
  case DrawType::imageMesh: // 4
    assert(!sbpLoad);
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

  bool isNotImageDraw = sbpLoad ? (drawType != DrawType::imageRect)
                                : (drawType != DrawType::imageMesh);

  const CullModeFlags cullMode =
      isNotImageDraw ? CullModeFlagBits::BackBit : CullModeFlagBits::None;

  return m_device.createGraphicsPipeline(GraphicsPipelineOptions{
      .shaderStages =
          {
              ShaderStage{
                  .shaderModule = vertexShader,
                  .stage = ShaderStageFlagBits::VertexBit,
              },
              ShaderStage{
                  .shaderModule = fragmentShader,
                  .stage = ShaderStageFlagBits::FragmentBit,
              },
          },
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
              .cullMode = cullMode,
              .frontFace = FrontFace::Clockwise,
          },
  });
}

// void PLSRenderContextKDGpuImpl::prepareToMapBuffers()

void PLSRenderTargetKDGpu::setTargetTextureView(
    KDGpu::Handle<KDGpu::TextureView_t> textureView) {
  m_targetTextureView = textureView;
}

/// this version of the function is a misnomer and operates on already compiled
/// shaders
static std::vector<uint32_t> charBufferToCode(const uint32_t *buf,
                                              size_t sizeInBytes) {
  std::vector<uint32_t> code;
  assert(sizeInBytes % sizeof(uint32_t) == 0);
  code.resize(sizeInBytes / sizeof(uint32_t));
  std::memcpy(code.data(), buf, sizeInBytes);
  return code;
}

static std::vector<uint32_t>
charBufferToCode(const char *buf, size_t size,
                 const char *identifier = "UNKNOWN_SHADER") {
  using namespace shaderc;
  Compiler shaderCompiler;
  SpvCompilationResult compiledShader = shaderCompiler.CompileGlslToSpv(
      buf, size, shaderc_shader_kind::shaderc_glsl_infer_from_source,
      identifier);

  SPDLOG_LOGGER_INFO(
      KDGpu::Logger::logger(),
      "ENCOUNTERED {} ERRORS AND {} WARNINGS WHILE COMPILING SHADER [ {} ]",
      compiledShader.GetNumErrors(), compiledShader.GetNumWarnings(),
      identifier);

  if (compiledShader.GetCompilationStatus() !=
      shaderc_compilation_status::shaderc_compilation_status_success) {
    SPDLOG_LOGGER_ERROR(KDGpu::Logger::logger(),
                        "SHADER COMPILATION FAILED: {}",
                        compiledShader.GetErrorMessage().c_str());
    return {};
  }
  return {compiledShader.cbegin(), compiledShader.cend()};
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
                .resourceType = ResourceBindingType::StorageBuffer,
                .shaderStages =
                    ShaderStageFlags(ShaderStageFlagBits::VertexBit),
            },
            {
                .binding = CONTOUR_BUFFER_IDX,
                .count = 1,
                .resourceType = ResourceBindingType::StorageBuffer,
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

    vertexShader = device.createShaderModule(
        charBufferToCode(tessellate_vert, sizeof(tessellate_vert)));

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

    if (plsType == PixelLocalStorageType::subpassLoad) {
      const char *language = "glsl";
      const char *versionString = "#version 460";

      std::ostringstream glsl;
      auto addDefine = [&glsl](const char *name) {
        glsl << "#define " << name << "\n";
      };
      glsl << "#extension GL_EXT_samplerless_texture_functions : enable\n";
      addDefine(GLSL_TARGET_VULKAN);
      addDefine(plsType == PixelLocalStorageType::subpassLoad
                    ? GLSL_PLS_IMPL_SUBPASS_LOAD
                    : GLSL_PLS_IMPL_NONE);

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
      vertexShader = context.device().createShaderModule(charBufferToCode(
          vertexGLSL.str().c_str(), vertexGLSL.str().size(),
          "disabled storage buffers draw pipeline vertex shader"));

      std::ostringstream fragmentGLSL;
      fragmentGLSL << versionString << "\n";
      fragmentGLSL << "#pragma shader_stage(fragment)\n";
      fragmentGLSL << "#define " GLSL_FRAGMENT "\n";
      fragmentGLSL << glsl.str();
      fragmentShader = context.device().createShaderModule(charBufferToCode(
          fragmentGLSL.str().c_str(), fragmentGLSL.str().size(),
          "disabled storage buffers draw pipeline fragment shader"));
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
    uint32_t height, KDGpu::TextureUsageFlags additionalTextureFlags,
    const PLSOptions::ContextOptions &options)
    : PLSRenderTarget(width, height), m_framebufferFormat(framebufferFormat) {
  using namespace KDGpu;
  KDGpu::TextureOptions desc{
      .type = TextureType::TextureType2D,
      .format = Format::R32_UINT,
      .extent = {.width = static_cast<uint32_t>(width),
                 .height = static_cast<uint32_t>(height),
                 .depth = 1},
      .mipLevels = 1,
      .usage =
          additionalTextureFlags | TextureUsageFlagBits::ColorAttachmentBit,
  };

  if (options.plsType == PLSOptions::PixelLocalStorageType::subpassLoad) {
    // desc.usage |= TextureUsageFlagBits::TransientAttachmentBit;
    desc.usage |= TextureUsageFlagBits::SampledBit;
  }

  desc.label = "coverageTexture";
  m_coverageTexture = device.createTexture(desc);
  desc.label = "clipTexture";
  m_clipTexture = device.createTexture(desc);

  desc.format = m_framebufferFormat;
  desc.label = "originalDstColorTexture";
  m_originalDstColorTexture = device.createTexture(desc);

  m_targetTextureView = {};
  m_coverageTextureView =
      m_coverageTexture.createView({.label = "coverageTextureView"});
  m_clipTextureView = m_clipTexture.createView({.label = "clipTextureView"});
  m_originalDstColorTextureView = m_originalDstColorTexture.createView(
      {.label = "originalDstColorTextureView"});
}

rcp<PLSRenderTargetKDGpu>
PLSRenderContextKDGpuImpl::makeRenderTarget(KDGpu::Format framebufferFormat,
                                            uint32_t width, uint32_t height) {
  using namespace KDGpu;
  return rcp(new PLSRenderTargetKDGpu(m_device, framebufferFormat, width,
                                      height, {}, m_contextOptions));
}

void PLSRenderContextKDGpuImpl::flush(const FlushDescriptor &desc) {
  using namespace KDGpu;
  auto *renderTarget =
      static_cast<const PLSRenderTargetKDGpu *>(desc.renderTarget);
  CommandRecorder commandRecorder = m_device.createCommandRecorder();

  // Render the complex color ramps to the gradient texture.
  if (desc.complexGradSpanCount > 0) {
    const auto &uniformBuffers =
        *static_cast<const BufferKDGpu *>(flushUniformBufferRing());

    const auto &gradSpanBuffers =
        *static_cast<const BufferKDGpu *>(gradSpanBufferRing());

    m_gradientBindings = m_device.createBindGroup(BindGroupOptions{
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
                .finalLayout = TextureLayout::TransferDstOptimal,
            }}});
    m_gradientTextureLayout = TextureLayout::TransferDstOptimal;

    gradPass.setViewport(Viewport{
        .x = 0.f,
        .y = static_cast<float>(desc.complexGradRowsTop),
        .width = static_cast<float>(pls::kGradTextureWidth),
        .height = static_cast<float>(desc.complexGradRowsTop),
    });

    gradPass.setPipeline(m_colorRampPipeline->renderPipeline());
    gradPass.setVertexBuffer(0, gradSpanBuffers.submittedBuffer());
    gradPass.setBindGroup(0, m_gradientBindings);
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
        .dstTextureLayout = TextureLayout::TransferDstOptimal,
        .regions = {
            BufferTextureCopyRegion{
                .bufferOffset = desc.simpleGradDataOffsetInBytes,
                .textureSubResource =
                    TextureSubresourceLayers{
                        .aspectMask = TextureAspectFlagBits::ColorBit,
                    },
                .textureExtent =
                    {
                        .width = desc.simpleGradTexelsWidth,
                        .height = desc.simpleGradTexelsHeight,
                        .depth = 1,
                    },
            },
        }});
  }

  // Tessellate all curves into vertices in the tessellation texture.
  if (desc.tessVertexSpanCount > 0) {
    const auto &pathBuffers =
        *static_cast<const BufferKDGpu *>(pathBufferRing());

    auto getPathBufferBinding = [&]() -> BindingResource {
      return StorageBufferBinding{
          .buffer = pathBuffers.submittedBuffer(),
          .offset =
              static_cast<uint32_t>(desc.firstPath * sizeof(pls::PathData)),
      };
    };

    const auto &contourBuffers =
        *static_cast<const BufferKDGpu *>(contourBufferRing());

    auto getContourBufferBinding = [&]() -> BindingResource {
      return StorageBufferBinding{
          .buffer = pathBuffers.submittedBuffer(),
          .offset =
              static_cast<uint32_t>(desc.firstPath * sizeof(pls::PathData)),
      };
    };

    m_tesselationBindings = m_device.createBindGroup(BindGroupOptions{
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
                        .finalLayout = TextureLayout::ShaderReadOnlyOptimal,
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
    tessPass.setBindGroup(0, m_tesselationBindings);
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

  auto checkLayoutIsReadyForDrawpass =
      [&commandRecorder](KDGpu::TextureLayout &textureLayout,
                         KDGpu::Texture &texture) -> void {
    if (textureLayout != TextureLayout::ShaderReadOnlyOptimal) {
      commandRecorder.textureMemoryBarrier(TextureMemoryBarrierOptions{
          .srcStages = PipelineStageFlags(PipelineStageFlagBit::AllGraphicsBit),
          .srcMask = AccessFlagBit::TransferWriteBit,
          .dstStages =
              PipelineStageFlags(PipelineStageFlagBit::FragmentShaderBit),
          .dstMask = AccessFlags(AccessFlagBit::ShaderReadBit),
          .oldLayout = textureLayout,
          .newLayout = TextureLayout::ShaderReadOnlyOptimal,
          .texture = texture,
          .range =
              {
                  .aspectMask = TextureAspectFlagBits::ColorBit,
                  .levelCount = 1,
              },
      });
      textureLayout = TextureLayout::ShaderReadOnlyOptimal;
    }
  };
  checkLayoutIsReadyForDrawpass(m_nullImagePaintTextureLayout,
                                m_nullImagePaintTexture);
  checkLayoutIsReadyForDrawpass(m_gradientTextureLayout, m_gradientTexture);

  RenderPassCommandRecorder drawPass =
      makePLSRenderPass(commandRecorder, *renderTarget, loadOp, clearColor);

  drawPass.setViewport(Viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<float>(renderTarget->width()),
      .height = static_cast<float>(renderTarget->height()),
  });

  if (m_contextOptions.plsType == PixelLocalStorageType::subpassLoad) {
    m_frameBindings.push_back(m_device.createBindGroup(BindGroupOptions{
        .layout = m_drawBindGroupLayouts[PLS_TEXTURE_BINDINGS_SET],
        .resources = {
            BindGroupEntry{
                .binding = FRAMEBUFFER_PLANE_IDX,
                .resource =
                    TextureViewBinding{renderTarget->m_targetTextureView},
            },
            BindGroupEntry{
                .binding = COVERAGE_PLANE_IDX,
                .resource =
                    TextureViewBinding{renderTarget->m_coverageTextureView},
            },
            BindGroupEntry{
                .binding = CLIP_PLANE_IDX,
                .resource = TextureViewBinding{renderTarget->m_clipTextureView},
            },
            BindGroupEntry{
                .binding = ORIGINAL_DST_COLOR_PLANE_IDX,
                .resource =
                    TextureViewBinding{
                        renderTarget->m_originalDstColorTextureView},
            },
        }}));

    drawPass.setBindGroup(PLS_TEXTURE_BINDINGS_SET, m_frameBindings.back());
  }

  const TextureView *currentImageTextureView = &m_nullImagePaintTextureView;
  bool needsNewBindings = true;
  // destroy bindings generated on previous  frame
  m_frameBindings.clear();

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
      m_frameBindings.push_back(
          m_device.createBindGroup(BindGroupOptions{
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

                  KDGpu::BindGroupEntry{
                      .binding = PATH_BUFFER_IDX,
                      .resource =
                          StorageBufferBinding{
                              .buffer = static_cast<const BufferKDGpu *>(
                                            pathBufferRing())
                                            ->submittedBuffer(),
                              .offset = static_cast<uint32_t>(
                                  desc.firstPath * sizeof(pls::PathData)),
                          },
                  },
                  KDGpu::BindGroupEntry{
                      .binding = PAINT_BUFFER_IDX,
                      .resource =
                          StorageBufferBinding{
                              .buffer = static_cast<const BufferKDGpu *>(
                                            paintBufferRing())
                                            ->submittedBuffer(),
                              .offset = static_cast<uint32_t>(
                                  desc.firstPaint * sizeof(pls::PaintData)),
                          },
                  },
                  KDGpu::BindGroupEntry{
                      .binding = PAINT_AUX_BUFFER_IDX,
                      .resource =
                          StorageBufferBinding{
                              .buffer = static_cast<const BufferKDGpu *>(
                                            paintAuxBufferRing())
                                            ->submittedBuffer(),
                              .offset = static_cast<uint32_t>(
                                  desc.firstPaintAux *
                                  sizeof(pls::PaintAuxData)),
                          },
                  },
                  KDGpu::BindGroupEntry{
                      .binding = CONTOUR_BUFFER_IDX,
                      .resource =
                          StorageBufferBinding{
                              .buffer = static_cast<const BufferKDGpu *>(
                                            contourBufferRing())
                                            ->submittedBuffer(),
                              .offset = static_cast<uint32_t>(
                                  desc.firstContour * sizeof(pls::ContourData)),
                          },
                  },
                  {
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
                  {
                      .binding = IMAGE_DRAW_UNIFORM_BUFFER_IDX,
                      // dynamic, determined by imageDrawDataOffset
                      .resource =
                          DynamicUniformBufferBinding{
                              .buffer = static_cast<const BufferKDGpu *>(
                                            imageDrawUniformBufferRing())
                                            ->submittedBuffer(),
                              .size = sizeof(pls::ImageDrawUniforms),
                          },
                  },
              }}));

      if (needsNewBindings || drawType == DrawType::imageRect ||
          drawType == DrawType::imageMesh) {
        drawPass.setBindGroup(0, m_frameBindings.back(), {},
                              {batch.imageDrawDataOffset});
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
      // wait for the target image to be available from swapchain acquisition
      .waitSemaphores = {swapchainImageAcquisitionCompletedSemaphore()},
      // signal that we are ready for presentation after submission
      .signalSemaphores = {renderToSwapchainImageCompletedSemaphore()},
      .signalFence = m_frameInFlightFence,
  });
}
} // namespace rive::pls
