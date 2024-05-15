#pragma once

#include "rive/pls/pls_render_context_helper_impl.hpp"
#include <KDGpu/device.h>
#include <map>

namespace rive::pls {

class PLSRenderTargetKDGpu : public PLSRenderTarget {
public:
  KDGpu::Format framebufferFormat() const { return m_framebufferFormat; }

  void setTargetTextureView(KDGpu::Handle<KDGpu::TextureView_t>);

private:
  friend class PLSRenderContextKDGpuImpl;
  // friend class PLSRenderContextWebGPUVulkan;

  PLSRenderTargetKDGpu(KDGpu::Device &device, KDGpu::Format framebufferFormat,
                       uint32_t width, uint32_t height,
                       KDGpu::TextureUsageFlags additionalTextureFlags);

  const KDGpu::Format m_framebufferFormat;

  KDGpu::Texture m_coverageTexture;
  KDGpu::Texture m_clipTexture;
  KDGpu::Texture m_originalDstColorTexture;

  KDGpu::Handle<KDGpu::TextureView_t> m_targetTextureView;
  KDGpu::TextureView m_coverageTextureView;
  KDGpu::TextureView m_clipTextureView;
  KDGpu::TextureView m_originalDstColorTextureView;
};

class PLSRenderContextKDGpuImpl : public PLSRenderContextHelperImpl {
public:
  enum class PixelLocalStorageType {
    // Pixel local storage cannot be supported; make a best reasonable effort to
    // draw shapes.
    none,
  };

  struct ContextOptions {
    bool disableStorageBuffers = false;
    PixelLocalStorageType plsType = PixelLocalStorageType::none;
  };

  virtual ~PLSRenderContextKDGpuImpl();

  /// Supported formats are B8G8R8A8_UNORM and R8G8B8A8_UNORM
  virtual rcp<PLSRenderTargetKDGpu>
  makeRenderTarget(KDGpu::Format, uint32_t width, uint32_t height);

  static std::unique_ptr<PLSRenderContext>
  MakeContext(KDGpu::Device &&, KDGpu::Queue &&, const ContextOptions &,
              const pls::PlatformFeatures &baselinePlatformFeatures = {});

  rcp<RenderBuffer> makeRenderBuffer(RenderBufferType, RenderBufferFlags,
                                     size_t) override;

  rcp<PLSTexture> makeImageTexture(uint32_t width, uint32_t height,
                                   uint32_t mipLevelCount,
                                   const uint8_t imageDataRGBA[]) override;

  KDGpu::Device &device() { return m_device; }
  const KDGpu::Device &device() const { return m_device; }

  inline void wait() { m_frameInFlightFence.wait(); }

  inline KDGpu::Handle<KDGpu::GpuSemaphore_t>
  swapchainImageAcquisitionCompletedSemaphore() {
    return m_imageAvailableSemaphore;
  }

  inline KDGpu::Handle<KDGpu::GpuSemaphore_t>
  renderToSwapchainImageCompletedSemaphore() {
    return m_renderCompleteSemaphore;
  }

protected:
  PLSRenderContextKDGpuImpl(
      KDGpu::Device &&, KDGpu::Queue &&, const ContextOptions &,
      const pls::PlatformFeatures &baselinePlatformFeatures);

  // Create a standard PLS "draw" pipeline for the current implementation.
  virtual KDGpu::GraphicsPipeline
  makePLSDrawPipeline(rive::pls::DrawType drawType,
                      KDGpu::Format framebufferFormat,
                      const KDGpu::ShaderModule &vertexShader,
                      const KDGpu::ShaderModule &fragmentShader);

  // Create a standard PLS "draw" render pass for the current implementation.
  virtual KDGpu::RenderPassCommandRecorder
  makePLSRenderPass(KDGpu::CommandRecorder &commandRecorder,
                    const PLSRenderTargetKDGpu &renderTarget,
                    KDGpu::AttachmentLoadOperation loadOp,
                    const KDGpu::ColorClearValue &clearColor);

  const KDGpu::PipelineLayout &drawPipelineLayout() const {
    return m_drawPipelineLayout;
  }

private:
  std::unique_ptr<BufferRing>
  makeUniformBufferRing(size_t capacityInBytes) override;
  std::unique_ptr<BufferRing>
  makeStorageBufferRing(size_t capacityInBytes,
                        pls::StorageBufferStructure) override;
  std::unique_ptr<BufferRing>
  makeVertexBufferRing(size_t capacityInBytes) override;
  std::unique_ptr<BufferRing>
  makeTextureTransferBufferRing(size_t capacityInBytes) override;

  void resizeGradientTexture(uint32_t width, uint32_t height) override;
  void resizeTessellationTexture(uint32_t width, uint32_t height) override;

  void prepareToMapBuffers() override {}

  void flush(const FlushDescriptor &) override;

  ContextOptions m_contextOptions;

  KDGpu::Device m_device;
  KDGpu::Queue m_queue;

  // Renders color ramps to the gradient texture.
  class ColorRampPipeline;
  std::unique_ptr<ColorRampPipeline> m_colorRampPipeline;
  KDGpu::Texture m_gradientTexture;
  KDGpu::TextureView m_gradientTextureView;

  // Renders tessellated vertices to the tessellation texture.
  class TessellatePipeline;
  std::unique_ptr<TessellatePipeline> m_tessellatePipeline;
  KDGpu::Buffer m_tessSpanIndexBuffer;
  KDGpu::Texture m_tesselationTexture;
  KDGpu::TextureView m_tesselationTextureView;

  class DrawPipeline;
  std::map<uint32_t, DrawPipeline> m_drawPipelines;
  std::array<KDGpu::BindGroupLayout, 2> m_drawBindGroupLayouts;
  KDGpu::Sampler m_linearSampler;
  KDGpu::Sampler m_mipmapSampler;
  KDGpu::BindGroup m_samplerBindings;
  KDGpu::PipelineLayout m_drawPipelineLayout;
  KDGpu::Buffer m_patchVertexBuffer;
  KDGpu::Buffer m_patchIndexBuffer;
  KDGpu::Texture m_nullImagePaintTexture;
  KDGpu::TextureView m_nullImagePaintTextureView;

  // sync primitives
  KDGpu::GpuSemaphore m_imageAvailableSemaphore;
  KDGpu::GpuSemaphore m_renderCompleteSemaphore;
  KDGpu::Fence m_frameInFlightFence;

  // per-frame resources
  KDGpu::CommandBuffer m_commandBuffer;
  KDGpu::BindGroup m_tesselationBindings;
  KDGpu::BindGroup m_gradientBindings;

  // caching resources (used per-frame)
  std::vector<KDGpu::BindGroup> m_frameBindings;
};

} // namespace rive::pls
