dofile('rive_build_config.lua')

dependencies = os.getenv('DEPENDENCIES')

rive = '../../'
rive_tess = '../../tess'
rive_skia = '../../skia'
skia = dependencies .. '/skia'

dofile(rive .. '/decoders/premake5_v2.lua')
if _OPTIONS.renderer == 'tess' then
    dofile(path.join(path.getabsolute(rive_tess) .. '/build', 'premake5_tess.lua'))
else
    -- tess renderer includes this
    dofile(path.join(path.getabsolute(rive) .. '/build', 'premake5.lua'))
end

dofile(path.join(path.getabsolute(rive) .. '/cg_renderer', 'premake5.lua'))

project('rive_viewer')
do
    dependson('rive_decoders')
    kind('ConsoleApp')

    defines({ 'WITH_RIVE_TEXT', 'WITH_RIVE_AUDIO', 'WITH_RIVE_LAYOUT', 'YOGA_EXPORT=' })

    includedirs({
        '../include',
        rive .. '/include',
        rive .. '/decoders/include',
        rive .. '/skia/renderer/include', -- for font backends
        dependencies,
        dependencies .. '/sokol',
        dependencies .. '/imgui',
        miniaudio,
        yoga,
    })

    links({ 'rive', 'rive_decoders', 'rive_harfbuzz', 'rive_sheenbidi', 'rive_yoga', 'miniaudio' })

    libdirs({ rive .. '/build/%{cfg.system}/bin/%{cfg.buildcfg}' })

    files({
        '../src/**.cpp',
        rive .. '/utils/**.cpp',
        dependencies .. '/imgui/imgui.cpp',
        dependencies .. '/imgui/imgui_widgets.cpp',
        dependencies .. '/imgui/imgui_tables.cpp',
        dependencies .. '/imgui/imgui_draw.cpp',
    })

    buildoptions({ '-Wall', '-fno-exceptions', '-fno-rtti' })

    filter({ 'system:macosx' })
    do
        links({
            'Cocoa.framework',
            'IOKit.framework',
            'CoreVideo.framework',
            'OpenGL.framework',
            'rive_cg_renderer',
        })
        files({ '../src/**.m', '../src/**.mm' })
    end

    filter({ 'system:macosx', 'options:graphics=gl' })
    do
        links({ 'OpenGL.framework' })
    end

    filter({ 'system:macosx', 'options:graphics=metal' })
    do
        links({ 'Metal.framework', 'MetalKit.framework', 'QuartzCore.framework' })
    end

    -- Tess Renderer Configuration
    filter({ 'options:renderer=tess' })
    do
        includedirs({ rive_tess .. '/include', rive .. '/decoders/include' })
        defines({ 'RIVE_RENDERER_TESS' })
        links({ 'rive_tess_renderer', 'rive_decoders', 'libpng', 'zlib', 'libjpeg', 'libwebp' })
        libdirs({ rive_tess .. '/build/%{cfg.system}/bin/%{cfg.buildcfg}' })
    end

    filter({ 'options:renderer=tess', 'options:graphics=gl' })
    do
        defines({ 'SOKOL_GLCORE33' })
    end

    filter({ 'options:renderer=tess', 'options:graphics=metal' })
    do
        defines({ 'SOKOL_METAL' })
    end

    filter({ 'options:renderer=tess', 'options:graphics=d3d' })
    do
        defines({ 'SOKOL_D3D11' })
    end

    filter({ 'options:renderer=skia', 'options:graphics=gl' })
    do
        defines({ 'SK_GL', 'SOKOL_GLCORE33' })
        files({ '../src/skia/viewer_skia_gl.cpp' })
        libdirs({ skia .. '/out/gl/%{cfg.buildcfg}' })
    end

    filter({ 'options:renderer=skia', 'options:graphics=metal' })
    do
        defines({ 'SK_METAL', 'SOKOL_METAL' })
        libdirs({ skia .. '/out/metal/%{cfg.buildcfg}' })
    end

    filter({ 'options:renderer=skia', 'options:graphics=d3d' })
    do
        defines({ 'SK_DIRECT3D' })
        libdirs({ skia .. '/out/d3d/%{cfg.buildcfg}' })
    end

    filter({ 'options:renderer=skia' })
    do
        includedirs({
            skia,
            skia .. '/include/core',
            skia .. '/include/effects',
            skia .. '/include/gpu',
            skia .. '/include/config',
        })
        defines({ 'RIVE_RENDERER_SKIA' })
        libdirs({
            rive_skia .. '/renderer/build/%{cfg.system}/bin/%{cfg.buildcfg}',
        })
        links({ 'skia', 'rive_skia_renderer' })
    end

    -- CLI config options
    newoption({
        trigger = 'graphics',
        value = 'gl',
        description = 'The graphics api to use.',
        allowed = { { 'gl' }, { 'metal' }, { 'd3d' } },
    })

    newoption({
        trigger = 'renderer',
        value = 'skia',
        description = 'The renderer to use.',
        allowed = { { 'skia' }, { 'tess' } },
    })
end
