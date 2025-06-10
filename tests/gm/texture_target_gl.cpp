/*
 * Copyright 2024 Rive
 */

#include "gm.hpp"
#include "gmutils.hpp"
#include "common/testing_window.hpp"
#include "rive/renderer/render_context.hpp"
#include "rive/renderer/rive_renderer.hpp"

#ifdef RIVE_TOOLS_NO_GL
namespace rive::gpu
{
class TextureRenderTargetGL;
};
using GLuint = uint32_t;
#else
#include "rive/renderer/gl/render_context_gl_impl.hpp"
#include "rive/renderer/gl/render_target_gl.hpp"
#endif

using namespace rivegm;
using namespace rive;
using namespace rive::gpu;

// Most gms render directly to the framebuffer. This GM checks that texture
// targets work in GL.
class TextureTargetGL : public GM
{
public:
    TextureTargetGL() : GM(256, 256) {}

    ColorInt clearColor() const override { return 0xffff0000; }

    void onDraw(rive::Renderer* originalRenderer) override
    {
#ifndef RIVE_TOOLS_NO_GL
        if (auto plsImplGL = TestingWindow::Get()->renderContextGLImpl())
        {
            // Intercept the current frameDescriptor and end the PLS frame.
            auto renderContext = TestingWindow::Get()->renderContext();
            auto originalFrameDescriptor = renderContext->frameDescriptor();
            auto originalRenderTarget = static_cast<RenderTargetGL*>(
                TestingWindow::Get()->renderTarget());
            TestingWindow::Get()->flushPLSContext();
            plsImplGL->unbindGLInternalResources();

            // Create an offscreen texture.
            if (m_offscreenTex == 0)
            {
                glGenTextures(1, &m_offscreenTex);
                glBindTexture(GL_TEXTURE_2D, m_offscreenTex);
                glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 256, 256);
                glTexParameteri(GL_TEXTURE_2D,
                                GL_TEXTURE_MIN_FILTER,
                                GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D,
                                GL_TEXTURE_MAG_FILTER,
                                GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D,
                                GL_TEXTURE_WRAP_S,
                                GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D,
                                GL_TEXTURE_WRAP_T,
                                GL_CLAMP_TO_EDGE);
            }
            auto textureTargetGL = make_rcp<TextureRenderTargetGL>(256, 256);
            textureTargetGL->setTargetTexture(m_offscreenTex);
            plsImplGL->invalidateGLState();

            // Draw to the offscreen texture.
            auto textureFrameDescriptor = originalFrameDescriptor;
            textureFrameDescriptor.clearColor = 0xffff00ff;
            renderContext->beginFrame(std::move(textureFrameDescriptor));
            RiveRenderer renderer(renderContext);
            drawInternal(&renderer, textureTargetGL.get());
            renderContext->flush({.renderTarget = textureTargetGL.get()});

            // Copy the offscreen texture back to the destination framebuffer.
            auto copyFrameDescriptor = originalFrameDescriptor;
            copyFrameDescriptor.loadAction =
                gpu::LoadAction::preserveRenderTarget;
            copyFrameDescriptor.clearColor = 0xffff0000;
            renderContext->beginFrame(std::move(copyFrameDescriptor));
            originalRenderTarget->bindDestinationFramebuffer(GL_FRAMEBUFFER);
            plsImplGL->blitTextureToFramebufferAsDraw(m_offscreenTex,
                                                      {0, 0, 256, 256},
                                                      256);
        }
        else
#endif
        {
            // We aren't PLS/GL, but still draw with a red background, just for
            // fun.
            drawInternal(originalRenderer, nullptr);
        }
    }

    virtual void drawInternal(Renderer* renderer, TextureRenderTargetGL*)
    {
        drawStar5(renderer, Paint(0x8000ffff));
        drawStar13(renderer, Paint(0x80ffff00));
    }

    void drawStar5(Renderer* renderer, RenderPaint* paint)
    {
        PathBuilder builder;
        float theta = -math::PI / 7;
        builder.moveTo(cosf(theta), sinf(theta));
        for (int i = 0; i <= 7; ++i)
        {
            theta += 2 * math::PI * 2 / 7;
            builder.lineTo(cosf(theta), sinf(theta));
        }
        renderer->save();
        renderer->translate(100, 100);
        renderer->scale(80, 80);
        renderer->drawPath(builder.detach(), paint);
        renderer->restore();
    }

    void drawStar13(Renderer* renderer, RenderPaint* paint)
    {
        PathBuilder builder;
        float theta = 0;
        for (int i = 0; i <= 13; ++i)
        {
            theta += 2 * math::PI * 3 / 13;
            builder.lineTo(cosf(theta), sinf(theta));
        }
        builder.fillRule(FillRule::evenOdd);
        renderer->save();
        renderer->translate(256 - 100, 256 - 100);
        renderer->scale(80, 80);
        renderer->drawPath(builder.detach(), paint);
        renderer->restore();
    }

    ~TextureTargetGL()
    {
#ifndef RIVE_TOOLS_NO_GL
        if (m_offscreenTex != 0)
        {
            glDeleteTextures(1, &m_offscreenTex);
        }
#endif
    }

private:
#ifndef RIVE_TOOLS_NO_GL
    GLuint m_offscreenTex = 0;
#endif
};

GMREGISTER(texture_target_gl, return new TextureTargetGL)

// This GM checks that texture targets (including MSAA targets) work with
// LoadAction::preserveRenderTarget.
class TextureTargetGLPreserve : public TextureTargetGL
{
public:
    TextureTargetGLPreserve(BlendMode blendMode) :
        TextureTargetGL(), m_blendMode(blendMode)
    {}
    TextureTargetGLPreserve() : TextureTargetGLPreserve(BlendMode::srcOver) {}

    virtual void drawInternal(Renderer* renderer,
                              TextureRenderTargetGL* renderTextureTargetGL)
    {
        ColorInt colors[2];
        float stops[2];
        colors[0] = 0xff000000;
        stops[0] = 0;
        colors[1] = 0xffff00ff;
        stops[1] = 1;
        Paint paint;
        paint->shader(TestingWindow::Get()
                          ->factory()
                          ->makeLinearGradient(0, 0, 250, 0, colors, stops, 2));
        renderer->drawPath(PathBuilder::Rect({0, 0, 256, 256}), paint);

        colors[0] = 0x80000000;
        stops[0] = 0;
        colors[1] = 0x8000ffff;
        stops[1] = 1;
        Paint paint2;
        paint2->shader(
            TestingWindow::Get()
                ->factory()
                ->makeLinearGradient(0, 0, 0, 250, colors, stops, 2));
        renderer->drawPath(PathBuilder::Rect({0, 0, 256, 256}), paint2);

        if (auto renderContext = TestingWindow::Get()->renderContext())
        {
            auto frameDescriptor = renderContext->frameDescriptor();
#ifndef RIVE_TOOLS_NO_GL
            if (auto plsImplGL = TestingWindow::Get()->renderContextGLImpl())
            {
                assert(renderTextureTargetGL);
                renderContext->flush({.renderTarget = renderTextureTargetGL});
                if (int sampleCount = frameDescriptor.msaaSampleCount)
                {
                    // If the MSAA framebuffer target is not the target texture,
                    // wipe it to red behind the scenes in order to make sure
                    // our preservation codepath works. (It shouldn't appear red
                    // in the end -- this should get preserved instead.)
                    if (!plsImplGL->capabilities()
                             .EXT_multisampled_render_to_texture)
                    {
                        renderTextureTargetGL->bindMSAAFramebuffer(plsImplGL,
                                                                   sampleCount,
                                                                   nullptr,
                                                                   nullptr);
                        glClearColor(1, 0, 0, 1);
                        glClear(GL_COLOR_BUFFER_BIT);
                    }
                }
            }
            else
#endif
            {
                TestingWindow::Get()->flushPLSContext();
            }
            frameDescriptor.loadAction = gpu::LoadAction::preserveRenderTarget;
            renderContext->beginFrame(std::move(frameDescriptor));
        }
        Paint paint5(0x8000ffff);
        paint5->blendMode(m_blendMode);
        drawStar5(renderer, paint5);

        if (auto renderContext = TestingWindow::Get()->renderContext())
        {
            auto frameDescriptor = renderContext->frameDescriptor();
#ifndef RIVE_TOOLS_NO_GL
            if (auto plsImplGL = TestingWindow::Get()->renderContextGLImpl())
            {
                assert(renderTextureTargetGL);
                renderContext->flush({.renderTarget = renderTextureTargetGL});
                if (int sampleCount = frameDescriptor.msaaSampleCount)
                {
                    // If the MSAA framebuffer target is not the target texture,
                    // wipe it to red behind the scenes in order to make sure
                    // our preservation codepath works. (It shouldn't appear red
                    // in the end -- this should get preserved instead.)
                    if (!plsImplGL->capabilities()
                             .EXT_multisampled_render_to_texture)
                    {
                        renderTextureTargetGL->bindMSAAFramebuffer(plsImplGL,
                                                                   sampleCount,
                                                                   nullptr,
                                                                   nullptr);
                        glClearColor(1, 0, 0, 1);
                        glClear(GL_COLOR_BUFFER_BIT);
                    }
                }
            }
            else
#endif
            {
                TestingWindow::Get()->flushPLSContext();
            }
            frameDescriptor.loadAction = gpu::LoadAction::preserveRenderTarget;
            renderContext->beginFrame(std::move(frameDescriptor));
        }
        Paint paint13(0x80ffff00);
        paint13->blendMode(m_blendMode);
        drawStar13(renderer, paint13);
    }

private:
    BlendMode m_blendMode;
};
GMREGISTER(texture_target_gl_preserve, return new TextureTargetGLPreserve)

// ...And verify that blend modes work on a texture target.
class TextureTargetGLPreserveLum : public TextureTargetGLPreserve
{
public:
    TextureTargetGLPreserveLum() :
        TextureTargetGLPreserve(BlendMode::luminosity)
    {}
};
GMREGISTER(texture_target_gl_preserve_lum,
           return new TextureTargetGLPreserveLum)
