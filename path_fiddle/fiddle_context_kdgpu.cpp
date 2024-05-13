#include "fiddle_context.hpp"

#include "path_fiddle.hpp"
#include "rive/pls/kdgpu/pls_render_context_kdgpu_impl.hpp"
#include "rive/pls/pls_renderer.hpp"

#include <KDGpu/buffer_options.h>
#include <KDGpu/graphics_api.h>
#include <KDGpu/instance.h>
#include <KDGpu/swapchain_options.h>
#include <KDGpu/vulkan/vulkan_graphics_api.h>

#define GLFW_INCLUDE_NONE
#if defined(KDGUI_PLATFORM_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(KDGUI_PLATFORM_XCB)
// TODO: see how to give kdgui xcb resources from x11 resources, if possible
#define GLFW_EXPOSE_NATIVE_X11
#elif defined(KDGUI_PLATFORM_WAYLAND)
#define GLFW_EXPOSE_NATIVE_WAYLAND
#elif defined(KDGUI_PLATFORM_COCOA)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include "GLFW/glfw3.h"

using namespace rive;
using namespace rive::pls;

KDGpu::SurfaceOptions getSurfaceOptionsFrom(GLFWwindow *window) {
  using namespace KDGpu;
  SurfaceOptions surfaceOptions{};
#if defined(KDGUI_PLATFORM_WIN32)
  surfaceOptions.hWnd = glfwGetWin32Window(window);
#endif
#if defined(KDGUI_PLATFORM_XCB)
  RIVE_UNREACHABLE();
#endif
#if defined(KDGUI_PLATFORM_WAYLAND)
  surfaceOptions.display = glfwGetWaylandDisplay();
  surfaceOptions.surface = glfwGetWaylandWindow(window);
#endif
#if defined(KDGUI_PLATFORM_COCOA)
  // TODO: macos
  RIVE_UNREACHABLE();
#endif
  RIVE_UNREACHABLE();
}

class FiddleContextKDGpu : public FiddleContext {
public:
  FiddleContextKDGpu();

  void begin(const PLSRenderContext::FrameDescriptor &frameDescriptor) override;

  void end(GLFWwindow *window, std::vector<uint8_t> *pixelData) final;

  void toggleZoomWindow() final;

  void flushPLSContext() override;

  std::unique_ptr<Renderer> makeRenderer(int width, int height) override;

  void onSizeChanged(GLFWwindow *window, int width, int height,
                     uint32_t sampleCount) override;

  rive::pls::PLSRenderTarget *plsRenderTargetOrNull() override;
  rive::pls::PLSRenderContext *plsContextOrNull() override;
  rive::Factory *factory() override;
  float dpiScale(GLFWwindow *) const override;

private:
  KDGpu::Device &device() {
    return m_plsContext->static_impl_cast<PLSRenderContextKDGpuImpl>()
        ->device();
  }

  void createSwapchain(int width, int height);

  // pipeline and rendering resources
  std::unique_ptr<PLSRenderContext> m_plsContext;
  rcp<PLSRenderTargetKDGpu> m_renderTarget;

  // swapchain
  std::optional<KDGpu::Extent2D> m_swapchainDimensions;
  KDGpu::Swapchain m_swapchain;
  std::vector<KDGpu::TextureView> m_swapchainViews;
  uint32_t m_currentImageIndex;

  // integration
  std::unique_ptr<KDGpu::GraphicsApi> m_api;
  KDGpu::Instance m_instance;
  std::optional<KDGpu::Surface> m_surface;
  KDGpu::Adapter *m_adapter;

  // we synchronously copy pixels from the screen into memory every frame
  std::optional<KDGpu::Buffer> m_pixelReadBuff;
  KDGpu::Fence m_pixelCopyFence;
};

FiddleContextKDGpu::FiddleContextKDGpu() {
  using namespace KDGpu;
  auto vkapi = std::make_unique<VulkanGraphicsApi>();
  // NOTE: no clue why this needs to be reinterpret cast?
  // it should be an implicit conversion
  // maybe its just my language server?
  static_assert(std::is_base_of_v<GraphicsApi, VulkanGraphicsApi>);
  m_api = std::unique_ptr<GraphicsApi>(
      reinterpret_cast<GraphicsApi *>(vkapi.get()));

  m_instance = m_api->createInstance(InstanceOptions{
      .applicationName = "path_fiddle",
  });

  m_adapter = m_instance.selectAdapter(AdapterDeviceType::Default);

  if (!m_adapter) {
    fprintf(stderr,
            "Failed to find an adapter! Please try another adapter type.\n");
    return;
  }

  Device device = m_adapter->createDevice();
  Queue queue = device.queues()[0];

  m_plsContext = PLSRenderContextKDGpuImpl::MakeContext(
      std::move(device), std::move(queue),
      PLSRenderContextKDGpuImpl::ContextOptions{}, PlatformFeatures{});

  Device &contextOwnedDevice =
      m_plsContext->static_impl_cast<PLSRenderContextKDGpuImpl>()->device();

  m_pixelCopyFence =
      contextOwnedDevice.createFence(FenceOptions{.createSignalled = true});
}

float FiddleContextKDGpu::dpiScale(GLFWwindow *) const {
  // NOTE: this is copied from FiddleContextGL, idk if the ifdef is necessary
#ifdef __APPLE__
  return 2;
#else
  return 1;
#endif
}

void FiddleContextKDGpu::begin(
    const PLSRenderContext::FrameDescriptor &frameDescriptor) {
  using namespace KDGpu;
  // find the texture view we can set as the render target for this frame
  m_currentImageIndex = 0;
  AcquireImageResult result = m_swapchain.getNextImageIndex(
      m_currentImageIndex,
      m_plsContext->static_impl_cast<PLSRenderContextKDGpuImpl>()
          ->waitSemaphore());

  if (result == AcquireImageResult::OutOfDate) {
    // This can happen when swapchain was resized
    // We need to recreate the swapchain and retry next frame
    assert(m_swapchainDimensions);
    createSwapchain(m_swapchainDimensions->width,
                    m_swapchainDimensions->height);
    return;
  }

  if (result != AcquireImageResult::Success) {
    fprintf(stderr, "Unable to acquire swapchain image");
    return;
  }

  // successfully acquired the image, set it as render target
  m_renderTarget->setTargetTextureView(
      m_swapchainViews.at(m_currentImageIndex));

  m_plsContext->beginFrame(std::move(frameDescriptor));
}

void FiddleContextKDGpu::end(GLFWwindow *window,
                             std::vector<uint8_t> *pixelData) {
  using namespace KDGpu;
  m_plsContext->flush(PLSRenderContext::FlushResources{
      .renderTarget = m_renderTarget.get(),
  });

  if (pixelData != nullptr) {
    m_pixelCopyFence.reset();
    // Read back pixels from the framebuffer!
    uint32_t w = m_renderTarget->width();
    uint32_t h = m_renderTarget->height();
    uint32_t rowBytesInReadBuff = math::round_up_to_multiple_of<256>(w * 4);

    // Create a buffer to receive the pixels.
    if (!m_pixelReadBuff) {
      m_pixelReadBuff = device().createBuffer(BufferOptions{
          .size = h * rowBytesInReadBuff,
          .usage = BufferUsageFlagBits::TransferDstBit,
          .memoryUsage = MemoryUsage::CpuToGpu,
      });
    }

    // Blit the framebuffer into m_pixelReadBuff.
    CommandRecorder commandRecorder = device().createCommandRecorder();

    commandRecorder.copyTextureToBuffer(TextureToBufferCopy{
        .srcTexture = m_swapchain.textures().at(m_currentImageIndex),
        .dstBuffer = *m_pixelReadBuff,
        .regions = {BufferTextureCopyRegion{
            .bufferOffset = 0,
            .bufferRowLength = rowBytesInReadBuff,
            .textureOffset = Offset3D{0, 0, 0},
            .textureExtent = Extent3D{.width = w, .height = h, .depth = 1},
        }},
    });

    CommandBuffer commands = commandRecorder.finish();

    device().queues()[0].submit(SubmitOptions{
        .commandBuffers = {commands},
        .signalFence = m_pixelCopyFence,
    });

    // Copy the image data from m_pixelReadBuff to pixelData.
    pixelData->resize(h * w * 4);

    // before we can map we need to wait for the copy to be complete
    m_pixelCopyFence.wait();
    void *mapping = m_pixelReadBuff->map();

    const uint8_t *pixelReadBuffData =
        reinterpret_cast<const uint8_t *>(mapping);
    for (size_t y = 0; y < h; ++y) {
      // Flip Y.
      const uint8_t *src = &pixelReadBuffData[(h - y - 1) * rowBytesInReadBuff];
      size_t row = y * w * 4;
      for (size_t x = 0; x < w * 4; x += 4) {
        // BGBRA -> RGBA.
        (*pixelData)[row + x + 0] = src[x + 2];
        (*pixelData)[row + x + 1] = src[x + 1];
        (*pixelData)[row + x + 2] = src[x + 0];
        (*pixelData)[row + x + 3] = src[x + 3];
      }
    }
    m_pixelReadBuff->unmap();
  }

  PLSRenderContextKDGpuImpl *const context =
      m_plsContext->static_impl_cast<PLSRenderContextKDGpuImpl>();

  device().queues()[0].present(KDGpu::PresentOptions{
      .waitSemaphores = {context->waitSemaphore()},
      .swapchainInfos = {{
          .swapchain = m_swapchain,
          .imageIndex = m_currentImageIndex,
      }},
  });

  context->wait();
}

void FiddleContextKDGpu::toggleZoomWindow() {}

void FiddleContextKDGpu::flushPLSContext() {
  m_plsContext->flush({.renderTarget = m_renderTarget.get()});
}

rive::Factory *FiddleContextKDGpu::factory() { return m_plsContext.get(); }

std::unique_ptr<Renderer> FiddleContextKDGpu::makeRenderer(int, int) {
  return std::make_unique<PLSRenderer>(m_plsContext.get());
}

void FiddleContextKDGpu::onSizeChanged(GLFWwindow *window, int width,
                                       int height, uint32_t sampleCount) {
  using namespace KDGpu;
  if (!m_surface)
    m_surface = m_instance.createSurface(getSurfaceOptionsFrom(window));

  createSwapchain(width, height);

  m_renderTarget =
      m_plsContext->static_impl_cast<PLSRenderContextKDGpuImpl>()
          ->makeRenderTarget(Format::R8G8B8A8_UNORM, width, height);
}

void FiddleContextKDGpu::createSwapchain(int width, int height) {
  using namespace KDGpu;
  // store the last dimensions for recreating swapchain at the same size
  m_swapchainDimensions = Extent2D{
      .width = static_cast<uint32_t>(width),
      .height = static_cast<uint32_t>(height),
  };
  const AdapterSwapchainProperties swapchainProperties =
      device().adapter()->swapchainProperties(*m_surface);
  const SurfaceCapabilities &surfaceCapabilities =
      swapchainProperties.capabilities;

  const Extent2D swapchainExtent = {
      .width =
          std::clamp((uint32_t)width, surfaceCapabilities.minImageExtent.width,
                     surfaceCapabilities.maxImageExtent.width),
      .height = std::clamp((uint32_t)height,
                           surfaceCapabilities.minImageExtent.height,
                           surfaceCapabilities.maxImageExtent.height),
  };

  const SwapchainOptions swapchainOptions{
      .surface = *m_surface,
      .minImageCount = getSuitableImageCount(swapchainProperties.capabilities),
      .imageExtent =
          {
              .width = swapchainExtent.width,
              .height = swapchainExtent.height,
          },
      .oldSwapchain = m_swapchain,
  };

  m_swapchain = device().createSwapchain(swapchainOptions);

  const auto &swapchainTextures = m_swapchain.textures();
  const auto swapchainTextureCount = swapchainTextures.size();

  m_swapchainViews.clear();
  m_swapchainViews.reserve(swapchainTextureCount);
  for (uint32_t i = 0; i < swapchainTextureCount; ++i) {
    auto view =
        swapchainTextures[i].createView({.format = swapchainOptions.format});
    m_swapchainViews.push_back(std::move(view));
  }
}

rive::pls::PLSRenderTarget *FiddleContextKDGpu::plsRenderTargetOrNull() {
  return m_renderTarget.get();
}

rive::pls::PLSRenderContext *FiddleContextKDGpu::plsContextOrNull() {
  return m_plsContext.get();
}

std::unique_ptr<FiddleContext> FiddleContext::MakeKDGpu() {
  static_assert(std::is_base_of_v<FiddleContext, FiddleContextKDGpu>);
  // TODO: why cant we static cast here, or implicit cast the unique ptr?
  return std::unique_ptr<FiddleContext>(
      reinterpret_cast<FiddleContext *>(new FiddleContextKDGpu()));
}
