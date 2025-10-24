import { mat3, vec2 } from 'gl-matrix';
// === 全局样式配置 ===
const sampleCount = 4; // 开启 4 倍 MSAA
let msaaTexture = null; //  提前声明全局变量
// 字符渲染样式
const STYLE = {
    charSize: 0.015,             // 字体宽（世界坐标）
    charWidthRatio: 1.5,         // 字体横向放大比例（>1 变宽）
    charHeightRatio: 1.5,        // 字体高宽比
    charSpacingRatio: 0.1,       // 字符之间的空隙（占charSize的比例）
    clearColor: [1.0, 1.0, 1.0, 0.1],//background
    charColor: [0.0, 0.0, 0.0, 0.9],  // 字体颜色

    // 节点矩形样式
    nodeColor: [0.6, 0.6, 0.6, 0.1],
    nodeColor_mini: [0.6, 0.6, 0.6, 1.0],

    // 连线样式
    edgeColor: [0.5, 0.5, 0.5, 0.5],
    edgeColor_mini: [0.5, 0.5, 0.5, 1.0],
    edgeCurveOffsetRatioX: 0.75, // X方向曲率：相对连线长度的偏移比例
    edgeCurveOffsetRatioY: 0.5, // Y方向曲率：相对连线长度的偏移比例
    edgeCurveSegments: 24,     // 采样段数：越大越平滑

    // 箭头样式
    arrowColor: [0.3, 0.3, 0.3, 1.0],
    arrowSize: 0.03,             // 箭头大小
    charShiftY: -0.05,
    ringColor: [0.6, 0.6, 0.6, 0.6],
    ringInnerColor: [0.6, 0.2, 0.8, 0.0],
    ringHighlightColor: [0.6, 0.6, 0.6, 0.9],

    viewRectColor: [0.0, 0.0, 1.0, 0.1],

    nodeBound: null,
    minimapMargin: 0.05,

    // === 新增暴露的 hover/selected 样式 ===
    rectHoverColor: [1.0, 0.0, 0.0, 0.0],
    charHoverColor: [0.2, 0.2, 1.0, 1.0],
    charSelectedColor: [1.0, 0.0, 0.0, 1.0],
    charNearestSwitchThreshold: 0.03,
    charCoverageGamma: 0.9,
    imageHoverTint: [1.0, 1.0, 0.0, 0.15],
    imageSelectedTint: [1.0, 0.0, 0.0, 0.25],
    ringHoverColor: [1.0, 0.8, 0.0, 0.8],
    ringSelectedColor: [1.0, 0.0, 0.0, 1.0],
    ringHoverGlowWidth: 0.08,
    ringSelectedGlowWidth: 0.12,
};
let signal = {
    mouseDownIdFlag: false,
    mouseDownID: [0, 0, 0],
    nodeMoveFlag: false,
    canvasMoveFlag: false,
    dragging: false,
}
let undoStack = []
let redoStack = []
let DataManager = {
    nodeColorIdMap: new Map(), // 节点颜色ID映射
    edgeColorIdMap: new Map(), // 边颜色ID映射
    adjGraph: new Map(), // 邻接图
    adjGraph_arrow: new Map(), // 邻接图, 用于存储箭头
    miniMapViewMatrix: null,
}
const commandDict = {
    CLICK: 1,
    DRAG: 2,
}

export async function initWebGPU(graph) {
    if (!navigator.gpu) {
        alert("WebGPU not supported");
        return;
    }




    const adapter = await navigator.gpu.requestAdapter();
    // const limits = adapter.limits;
    // console.log("maxSampleCount:", limits.maxSampledTexturesPerShaderStage);
    const device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", e => {
        console.error("GPU ERROR:", e.error);
    });
    const canvas = document.getElementById("webgpuCanvas");

    // 小地图 Canvas
    const miniMapCanvas = document.getElementById("minimapCanvas");
    const miniMapContext = miniMapCanvas.getContext("webgpu");
    // 配置小地图 context
    const miniMapFormat = navigator.gpu.getPreferredCanvasFormat();
    miniMapContext.configure({
        device,
        format: miniMapFormat,
        alphaMode: "opaque"
    });

    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({ device, format, alphaMode: "opaque" });
    //↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑加载全局资源


    const pickTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const chars = collectCharsFromGraph(graph)
    const { texture: fontTexture, samplerLinear: fontSampler, samplerNearest: fontSamplerNearest, uvMap } = await generateDigitTexture(device, chars);
    const imageList = ["../data/1.png"];
    const { canvas: imageCanvas, uvMap: imageUVMap } = await generateImageAtlas(imageList);
    const imageTexture = uploadTextureByImageData(device, imageCanvas, { enableSRGB: true, generateMipmaps: true });
    const imageLevelCount = Math.floor(Math.log2(Math.max(imageCanvas.width, imageCanvas.height))) + 1;
    const imageSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
        lodMinClamp: 0,
        lodMaxClamp: imageLevelCount - 1
    });



    let data = extractDataFromG6(graph, canvas, uvMap, imageUVMap);

    //↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑图元需要的额外资源

    // === Uniforms
    const viewMatrix = mat3.create();
    const uniformBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bounds = getGraphBounds(graph.nodes);  // 获取坐标范围
    DataManager.miniMapViewMatrix = getEquivalentNDCMatrix(bounds, STYLE.minimapMargin);

    const miniMapUniformBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // 写入单位矩阵





    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} },  // uniform
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // fontTex
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // fontSamp
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // imageTex
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },  // imageSamp
            { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} }   // fontSampNearest
        ]
    });



    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout] // group(0), group(1)
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: fontTexture.createView() },
            { binding: 2, resource: fontSampler },
            { binding: 3, resource: imageTexture.createView() },
            { binding: 4, resource: imageSampler },
            { binding: 5, resource: fontSamplerNearest }
        ]
    });

    const miniMapBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: miniMapUniformBuffer } },
            { binding: 1, resource: fontTexture.createView() },
            { binding: 2, resource: fontSampler },
            { binding: 3, resource: imageTexture.createView() },
            { binding: 4, resource: imageSampler },
            { binding: 5, resource: fontSamplerNearest }
        ]
    });

    //↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑绑定组布局
    // === Shader
    const shaderModule = device.createShaderModule({
        code: `
        
      struct Uniforms {
        viewMatrix : mat4x4<f32>,
          hoverPos1 : vec2<f32>,
          hoverPos2 : vec2<f32>
      };
    @group(0) @binding(0) var<uniform> uniforms : Uniforms;
    @group(0) @binding(1) var fontTex : texture_2d<f32>;
    @group(0) @binding(2) var fontSamp : sampler;
    @group(0) @binding(3) var imageTex : texture_2d<f32>;
    @group(0) @binding(4) var imageSamp : sampler;
    @group(0) @binding(5) var fontSampNearest : sampler;


      struct Out {
        @builtin(position) position : vec4<f32>,
        @location(0) color : vec4<f32>,
        @location(1) uv : vec2<f32>,
        @location(2) strokeWidth : f32,
        @location(3) isHighlight: f32, // 1.0 表示高亮，0.0 表示普通
        @location(4) selected: f32,
      };

      struct VertexIn {
        @location(0) pos: vec2<f32>,
        @location(1) instPos: vec2<f32>,
        @location(2) instSize: vec2<f32>,
        @location(3) instColor: vec4<f32>,
        @location(4) pickColor: vec4<f32>,
        @location(5) selected: f32
      };

      struct SimpleIn {
        @location(0) pos: vec2<f32>,
  @location(1) idColor: vec4<f32>
      };

      struct ArrowIn {
        @location(0) pos: vec2<f32>,
        @location(1) p1: vec2<f32>,
        @location(2) p2: vec2<f32>,
        @location(3) sourceColorId: vec4<f32>,
        @location(4) targetColorId: vec4<f32>
      };

      struct CharIn {
        @location(0) pos: vec2<f32>,
        @location(1) center: vec2<f32>,
        @location(2) u0: f32,
        @location(3) u1: f32,
        @location(4) v0: f32,
        @location(5) v1: f32,
        @location(6) pickColor: vec4<f32>,
        @location(7) selected: f32
      };
      struct ImageIn {
        @location(0) pos: vec2<f32>,
        @location(1) instPos: vec2<f32>,
        @location(2) instSize: vec2<f32>,
        @location(3) u0: f32,
        @location(4) u1: f32,
        @location(5) v0: f32,
        @location(6) v1: f32,
        @location(7) pickColor: vec4<f32>,
        @location(8) selected: f32
        };
struct RingIn {
  @location(0) pos: vec2<f32>,
  @location(1) center: vec2<f32>,
  @location(2) radius: f32,
  @location(3) strokeWidth: f32,
        @location(4) pickColor: vec4<f32>,
        @location(5) selected: f32
};

fn distanceSquared(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return dx * dx + dy * dy;
}
  fn crossProduct(o: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}
    fn isIntersecting(p1: vec2<f32>, p2: vec2<f32>, q1: vec2<f32>, q2: vec2<f32>) -> bool {
    // 检查线段 (p1, p2) 是否与线段 (q1, q2) 相交
    let c1 = crossProduct(q1, q2, p1);
    let c2 = crossProduct(q1, q2, p2);
    let c3 = crossProduct(p1, p2, q1);
    let c4 = crossProduct(p1, p2, q2);
    
    return (c1 * c2 < 0.0) && (c3 * c4 < 0.0);
}
@vertex
fn rect_vertex(input: VertexIn) -> Out {
  var out: Out;
  let world = input.instPos + input.pos * input.instSize;
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);

  // 节点矩形边界
  let left   = input.instPos.x - input.instSize.x / 2.0;
  let right  = input.instPos.x + input.instSize.x / 2.0;
  let top    = input.instPos.y + input.instSize.y / 2.0;
  let bottom = input.instPos.y - input.instSize.y / 2.0;

  // Hover 区域边界
  let hoverLeft   = uniforms.hoverPos1.x;
  let hoverTop    = uniforms.hoverPos1.y;
  let hoverRight  = uniforms.hoverPos2.x;
  let hoverBottom = uniforms.hoverPos2.y;

  // 矩形相交判断
  let isIntersecting =
      right >= hoverLeft &&
      left <= hoverRight &&
      top >= hoverBottom &&
      bottom <= hoverTop;

  // 着色逻辑
  if (isIntersecting) {
    out.color = vec4<f32>(${STYLE.rectHoverColor.join(", ")});
  } else {
    out.color = input.instColor;
  }

  return out;
}


      @vertex
      fn simple_vertex(input: SimpleIn) -> Out {
        var out: Out;
        out.position = uniforms.viewMatrix * vec4<f32>(input.pos, 0.0, 1.0);
        let edgeColor = vec4<f32>(${STYLE.edgeColor.join(", ")});
        out.color = edgeColor;
        return out;
      }

      @vertex
      fn arrow_vertex(input: ArrowIn) -> Out {
        var out: Out;
        let dir = input.p2 - input.p1;
        let angle = -atan2(dir.y, dir.x);
        let rot = mat2x2<f32>(
          cos(angle), -sin(angle),
          sin(angle),  cos(angle)
        );
        let world = input.p2 + rot * input.pos;
        out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);
        let arrowColor = vec4<f32>(${STYLE.arrowColor.join(", ")});
        out.color = arrowColor;
        return out;
      }

@vertex
fn char_vertex(input: CharIn) -> Out {
  var out: Out;
  let size = vec2<f32>(${STYLE.charSize * STYLE.charWidthRatio}, ${STYLE.charSize * STYLE.charHeightRatio});
  let offset = input.pos * size;
  let world = input.center + offset;
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);

  out.uv = vec2<f32>(
    mix(input.u0, input.u1, input.pos.x * 0.5 + 0.5),
    mix(input.v0, input.v1, input.pos.y * -0.5 + 0.5)
  );

  // 计算字符矩形与 hover 区是否相交
  let halfW = size.x * 0.5;
  let halfH = size.y * 0.5;
  let left   = input.center.x - halfW;
  let right  = input.center.x + halfW;
  let top    = input.center.y + halfH;
  let bottom = input.center.y - halfH;

  let hoverLeft   = uniforms.hoverPos1.x;
  let hoverTop    = uniforms.hoverPos1.y;
  let hoverRight  = uniforms.hoverPos2.x;
  let hoverBottom = uniforms.hoverPos2.y;

  let isIntersecting =
      right >= hoverLeft &&
      left <= hoverRight &&
      top >= hoverBottom &&
      bottom <= hoverTop;

  out.isHighlight = select(0.0, 1.0, isIntersecting);
  out.selected = input.selected;
  return out;
}

@vertex
fn ring_pick_vertex(input: RingIn) -> Out {
  var out: Out;
  let world = input.center + input.pos * vec2<f32>(input.radius);
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);
  out.color = input.pickColor;
  return out;
}

@vertex
fn ring_vertex(input: RingIn) -> Out {
  var out: Out;
  let world = input.center + input.pos * vec2<f32>(input.radius);
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);
  out.uv = input.pos;
  out.strokeWidth = input.strokeWidth;

  // === hover 判断（圆心到 hover 矩形的最近点距离） ===
  let cx = input.center.x;
  let cy = input.center.y;
  let r  = input.radius;

  let hoverLeft   = uniforms.hoverPos1.x;
  let hoverRight  = uniforms.hoverPos2.x;
  let hoverTop    = uniforms.hoverPos1.y;
  let hoverBottom = uniforms.hoverPos2.y;

  let closestX = clamp(cx, hoverLeft, hoverRight);
  let closestY = clamp(cy, hoverBottom, hoverTop);
  let distSq = (cx - closestX) * (cx - closestX) + (cy - closestY) * (cy - closestY);
  let isHover = distSq <= r * r / 4;

  let baseColor = vec4<f32>(${STYLE.ringColor.join(", ")});

  out.color = baseColor;
  out.isHighlight = select(0.0, 1.0, isHover);
  out.selected = input.selected;
  return out;
}


        @fragment
        fn fragment_main(input: Out) -> @location(0) vec4<f32> {
        return input.color; // 用于矩形、边、箭头
        }

        @fragment
        fn fragment_mini(input: Out) -> @location(0) vec4<f32> {
        return input.color; // 用于小地图矩形、边、箭头
        }

        @fragment
        fn char_frag(input: Out) -> @location(0) vec4<f32> {
        let texLinear = textureSample(fontTex, fontSamp, input.uv);
        let texNearest = textureSample(fontTex, fontSampNearest, input.uv);

        let w = fwidth(texLinear.a);
        let t0 = ${STYLE.charNearestSwitchThreshold};
        let t1 = ${STYLE.charNearestSwitchThreshold} * 2.0;
        let nearestWeight = 1.0 - smoothstep(t0, t1, w);
        let alphaSample = mix(texLinear.a, texNearest.a, nearestWeight);

        let wAdj = max(w, 1e-3);
        var coverage = smoothstep(0.5 - 0.5 * wAdj, 0.5 + 0.5 * wAdj, alphaSample);
        coverage = pow(coverage, ${STYLE.charCoverageGamma});

        let baseColor = vec4<f32>(${STYLE.charColor.join(", ")});
        let hoverColor = vec4<f32>(${STYLE.charHoverColor.join(", ")});
        let selectedColor = vec4<f32>(${STYLE.charSelectedColor.join(", ")});

        let hoverAlpha = input.isHighlight;
        let selectedAlpha = select(0.0, 1.0, input.selected > 0.0);

        var finalColor = baseColor;
        finalColor = mix(finalColor, hoverColor, hoverAlpha);
        finalColor = mix(finalColor, selectedColor, selectedAlpha);

        return vec4<f32>(finalColor.rgb, finalColor.a * coverage);
        }

        @vertex
fn image_vertex(input: ImageIn) -> Out {
  var out: Out;
  let world = input.instPos + input.pos * input.instSize;
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);
  out.uv = vec2<f32>(
    mix(input.u0, input.u1, input.pos.x + 0.5),
    mix(input.v1, input.v0, input.pos.y + 0.5)
  );
  out.color = vec4<f32>(1.0);

  // 计算图片矩形与 hover 区是否相交
  let left   = input.instPos.x - input.instSize.x / 2.0;
  let right  = input.instPos.x + input.instSize.x / 2.0;
  let top    = input.instPos.y + input.instSize.y / 2.0;
  let bottom = input.instPos.y - input.instSize.y / 2.0;

  let hoverLeft   = uniforms.hoverPos1.x;
  let hoverTop    = uniforms.hoverPos1.y;
  let hoverRight  = uniforms.hoverPos2.x;
  let hoverBottom = uniforms.hoverPos2.y;

  let isIntersecting =
      right >= hoverLeft &&
      left <= hoverRight &&
      top >= hoverBottom &&
      bottom <= hoverTop;

  out.isHighlight = select(0.0, 1.0, isIntersecting);
  out.selected = input.selected;

  return out;
}
@fragment
fn image_frag(input: Out) -> @location(0) vec4<f32> {
  let c = textureSample(imageTex, imageSamp, input.uv);
  let hoverTint = vec4<f32>(${STYLE.imageHoverTint.join(", ")});
  let selectedTint = vec4<f32>(${STYLE.imageSelectedTint.join(", ")});
  let hoverAlpha = input.isHighlight * hoverTint.a;
  let selectedAlpha = select(0.0, selectedTint.a, input.selected > 0.0);

  let rgbHover = mix(c.rgb, hoverTint.rgb, hoverAlpha);
  let rgbSelected = mix(rgbHover, selectedTint.rgb, selectedAlpha);

  return vec4<f32>(rgbSelected, c.a);
} 

@fragment
fn ring_frag(input: Out) -> @location(0) vec4<f32> {
  let r = length(input.uv);
  let outer = 0.4;
  let inner = outer - outer * input.strokeWidth;
  let edgeWidth = 0.01;

  // 提前裁剪到最大高亮带之外
  if (r > outer + max(${STYLE.ringHoverGlowWidth}, ${STYLE.ringSelectedGlowWidth}) + edgeWidth) {
    discard;
  }

  // 主环区域
  let tInner = smoothstep(inner - edgeWidth, inner + edgeWidth, r);
  let tOuter = 1.0 - smoothstep(outer - edgeWidth, outer + edgeWidth, r);
  let ringAlpha = tInner * tOuter;
  let baseColor = input.color;
  let baseAlpha = ringAlpha * baseColor.a;

  // 外部 hover / selected 模糊带
  let hoverColor = vec4<f32>(${STYLE.ringHoverColor.join(", ")});
  let selectedColor = vec4<f32>(${STYLE.ringSelectedColor.join(", ")});
  let tHover = smoothstep(outer, outer + ${STYLE.ringHoverGlowWidth}, r);
  let tSelected = smoothstep(outer, outer + ${STYLE.ringSelectedGlowWidth}, r);
  let hoverAlpha = (1.0 - tHover) * input.isHighlight;
  let selectedAlpha = (1.0 - tSelected) * select(0.0, 1.0, input.selected > 0.0);

  // 合成
  var finalColor = baseColor;
  var finalAlpha = baseAlpha;
  if (r >= inner) {
    let mixedHoverColor = mix(finalColor, hoverColor, hoverAlpha);
    let mixedHoverAlpha = max(finalAlpha, hoverAlpha * hoverColor.a);
    finalColor = mix(mixedHoverColor, selectedColor, selectedAlpha);
    finalAlpha = max(mixedHoverAlpha, selectedAlpha * selectedColor.a);
  }

  return vec4<f32>(finalColor.rgb, finalAlpha);
}



@fragment
fn ring_pick_frag(input: Out) -> @location(0) vec4<f32> {
  return input.color; // color 在 vertex shader 中写入 pickColor
}





    `
    });

    // === 创建各类 buffer（矩形、线段、箭头、字符）
    const imageInstanceBuffer = createBuffer(device, data.imageInstances, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,);
    const ringInstanceBuffer = createBuffer(device, data.ringInstances, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,);


    const quad = new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
    const quadBuffer = createBuffer(device, quad, GPUBufferUsage.VERTEX);

    const rectBuffer = createBuffer(device, data.rects, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,);
    const lineBuffer = createBuffer(device, data.polylines, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    const arrowVertexBuffer = createBuffer(device, new Float32Array([
        0.0, 0.0,
        -STYLE.arrowSize, STYLE.arrowSize * 0.4,
        -STYLE.arrowSize, -STYLE.arrowSize * 0.4
    ]), GPUBufferUsage.VERTEX);
    const arrowInstanceBuffer = createBuffer(device, data.arrowSegments, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    const charQuadBuffer = createBuffer(device, new Float32Array([
        -0.5, -0.5, 0.5, -0.5,
        -0.5, 0.5, 0.5, 0.5
    ]), GPUBufferUsage.VERTEX);
    const charInstanceBuffer = createBuffer(device, data.charData, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,);
    console.log("charData length:", data.charData.length);
    console.log("charData example:", data.charData.slice(0, 12));
    const expectedBytes = data.charData.length * Float32Array.BYTES_PER_ELEMENT;
    console.log("charInstanceBuffer size:", expectedBytes);

    const rectBuffer_mini = createBuffer(device, data.rects_mini, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,);
    const lineBuffer_mini = createBuffer(device, data.polylines_mini, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);





    const viewRect = getViewRect(canvas, viewMatrix);

    const viewRectBuffer = createBuffer(device, viewRect, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);


    // === 创建 pipelines（矩形、边、箭头、字符）
    /*
    const rectPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "rect_vertex", "fragment_main", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 52, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32x2", offset: 8 },
                { shaderLocation: 3, format: "float32x4", offset: 16 },
                { shaderLocation: 4, format: "float32x4", offset: 32 },
                { shaderLocation: 5, format: "float32", offset: 48 }
            ]
        }
    ], "triangle-strip", sampleCount);
    */

    const linePipeline = createPipeline(device, shaderModule, pipelineLayout, format, "simple_vertex", "fragment_main", [
        {
            arrayStride: 24, // 6 float × 4 bytes = 32
            stepMode: "vertex",
            attributes: [
                { shaderLocation: 0, format: "float32x2", offset: 0 },   // position
                { shaderLocation: 1, format: "float32x4", offset: 8 }    // colorID
            ]
        }
    ], "line-list", sampleCount);

    const arrowPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "arrow_vertex", "fragment_main", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 48, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32x2", offset: 8 },
                { shaderLocation: 3, format: "float32x4", offset: 16 },
                { shaderLocation: 4, format: "float32x4", offset: 32 }
            ]
        }
    ], "triangle-list", sampleCount);

    const charPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "char_vertex", "char_frag", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 44, // ✅ 每个实例数据共 6 个 float：2 + 4 = 24 字节
            stepMode: "instance",
            attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },   // center
                { shaderLocation: 2, format: "float32", offset: 8 },   // u0
                { shaderLocation: 3, format: "float32", offset: 12 },  // u1
                { shaderLocation: 4, format: "float32", offset: 16 },  // v0
                { shaderLocation: 5, format: "float32", offset: 20 },   // v1
                { shaderLocation: 6, format: "float32x4", offset: 24 },
                { shaderLocation: 7, format: "float32", offset: 40 }
            ]
        }
    ], "triangle-strip", sampleCount);
    const imagePipeline = createPipeline(device, shaderModule, pipelineLayout, format, "image_vertex", "image_frag", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 52, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32x2", offset: 8 },
                { shaderLocation: 3, format: "float32", offset: 16 },
                { shaderLocation: 4, format: "float32", offset: 20 },
                { shaderLocation: 5, format: "float32", offset: 24 },
                { shaderLocation: 6, format: "float32", offset: 28 },
                { shaderLocation: 7, format: "float32x4", offset: 32 },
                { shaderLocation: 8, format: "float32", offset: 48 }
            ]
        }
    ], "triangle-strip", sampleCount);
    const ringPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "ring_vertex", "ring_frag", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 36, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32", offset: 8 },
                { shaderLocation: 3, format: "float32", offset: 12 },
                { shaderLocation: 4, format: "float32x4", offset: 16 },
                { shaderLocation: 5, format: "float32", offset: 32 }
            ]
        }
    ], "triangle-strip", sampleCount);

    const miniMapRectPipeline = createPipeline(
        device,
        shaderModule,
        pipelineLayout,
        format,
        "rect_vertex",
        "fragment_mini",
        [
            { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
            {
                arrayStride: 52, stepMode: "instance", attributes: [
                    { shaderLocation: 1, format: "float32x2", offset: 0 },
                    { shaderLocation: 2, format: "float32x2", offset: 8 },
                    { shaderLocation: 3, format: "float32x4", offset: 16 },
                    { shaderLocation: 4, format: "float32x4", offset: 32 },
                    { shaderLocation: 5, format: "float32", offset: 48 }
                ]
            }
        ],
        "triangle-strip",
        1  // ✅ 使用 sampleCount: 1
    );
    const miniMapLinePipeline = createPipeline(device, shaderModule, pipelineLayout, format, "simple_vertex", "fragment_mini", [
        {
            arrayStride: 24, // 6 float × 4 bytes = 32
            stepMode: "vertex",
            attributes: [
                { shaderLocation: 0, format: "float32x2", offset: 0 },   // position
                { shaderLocation: 1, format: "float32x4", offset: 8 }    // colorID
            ]
        }
    ], "line-list", 1);

    // === 控制视图变换
    let scale = 1;
    let offset = [0, 0];
    let hoverPos1 = [999, 999]; // 默认值设置为图外
    let hoverPos2 = [999, 999]; // 默认值设置为图外
    const hoverRadius = 0.03;
    const updateMatrix = () => {
        mat3.identity(viewMatrix);
        mat3.scale(viewMatrix, viewMatrix, [scale, scale]);
        mat3.translate(viewMatrix, viewMatrix, offset);
        const mat4 = new Float32Array([
            viewMatrix[0], viewMatrix[1], 0, 0,
            viewMatrix[3], viewMatrix[4], 0, 0,
            0, 0, 1, 0,
            viewMatrix[6], viewMatrix[7], 0, 1,
            hoverPos1[0], hoverPos1[1], hoverPos2[0], hoverPos2[1] // 更新为两个坐标
        ]);
        // console.log(mat4);

        device.queue.writeBuffer(uniformBuffer, 0, mat4);
    };
    const updateMiniMapMatrix = () => {
        const mat4 = new Float32Array([
            DataManager.miniMapViewMatrix[0], DataManager.miniMapViewMatrix[1], 0, 0,
            DataManager.miniMapViewMatrix[3], DataManager.miniMapViewMatrix[4], 0, 0,
            0, 0, 1, 0,
            DataManager.miniMapViewMatrix[6], DataManager.miniMapViewMatrix[7], 0, 1,
            0, 0, 0, 0  // 小地图通常不用 hoverPos，先填 0
        ]);
        // const mat4 = new Float32Array([
        //     1, 0, 0, 0,
        //     0, 1, 0, 0,
        //     0, 0, 1, 0,
        //     0, 0, 0, 1,
        //     0, 0, 0, 0  // 小地图通常不用 hoverPos，先填 0
        // ]);
        device.queue.writeBuffer(miniMapUniformBuffer, 0, mat4);
    }


    // === 交互
    const hoverSize = 0.01;
    canvas.addEventListener("wheel", e => {
        e.preventDefault();

        const zoomStep = 0.1 // 动态步长（随scale增大而减小）
        let newScale = e.deltaY < 0 ? scale * (1 + zoomStep) : scale * (1 - zoomStep);
        const worldXbefore = (hoverPos1[0] + hoverPos2[0]) / 2
        const worldYbefore = (hoverPos1[1] + hoverPos2[1]) / 2
        console.log("before", worldXbefore, worldYbefore);
        console.log(offset);


        // offset[0] -= (worldXbefore - offset[0]) * (newScale / scale - 1);
        // offset[1] -= (worldYbefore - offset[1]) * (newScale / scale - 1);
        scale = newScale;

        updateMatrix();
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width * 2 - 1;
        const y = 1 - (e.clientY - rect.top) / canvas.height * 2;
        device.queue.writeBuffer(viewRectBuffer, 0, getViewRect(canvas, viewMatrix));
    });
    miniMapCanvas.addEventListener("wheel", e => {
        e.preventDefault();

        const zoomStep = 0.1 // 动态步长（随scale增大而减小）
        let newScale = e.deltaY < 0 ? scale * (1 - zoomStep) : scale * (1 + zoomStep);
        const worldXbefore = (hoverPos1[0] + hoverPos2[0]) / 2
        const worldYbefore = (hoverPos1[1] + hoverPos2[1]) / 2
        console.log("before", worldXbefore, worldYbefore);
        console.log(offset);


        // offset[0] -= (worldXbefore - offset[0]) * (newScale / scale - 1);
        // offset[1] -= (worldYbefore - offset[1]) * (newScale / scale - 1);
        scale = newScale;

        updateMatrix();
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width * 2 - 1;
        const y = 1 - (e.clientY - rect.top) / canvas.height * 2;
        device.queue.writeBuffer(viewRectBuffer, 0, getViewRect(canvas, viewMatrix));
    });

    canvas.addEventListener("click", async e => {
        console.log("click1");

        // if (signal.mouseDownIdFlag) { console.log(13131); return }
        if (signal.nodeMoveFlag || signal.canvasMoveFlag) {
            signal.nodeMoveFlag = false;
            signal.canvasMoveFlag = false;
            return
        }

        // 背景点击时：清除所有选中状态
        if (matchColorID(signal.mouseDownID, [0, 0, 0])) {
            clearAllSelections(data);
            device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);
            device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);
            device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);
            device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);

            signal.mouseDownIdFlag = false; // 重置鼠标点击 ID
            console.log("cleared selection by background click");
            return;
        }

        const select = true
        undoStack.push({
            type: commandDict.CLICK,
            config: signal.mouseDownID
        });
        redoStack = [];

        markSelectedById(signal.mouseDownID, data.rects, 13, -1, select);
        device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

        markSelectedById(signal.mouseDownID, data.ringInstances, 9, -1, select);
        device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

        markSelectedById(signal.mouseDownID, data.imageInstances, 13, -1, select);
        device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

        markSelectedById(signal.mouseDownID, data.charData, 11, -1, select);
        device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);

        signal.mouseDownIdFlag = false; // 重置鼠标点击 ID

        console.log("click2");
    });

    // 清除所有实例的选中位（最后一位）
    function clearAllSelections(data) {
        const clear = (arr, stride) => {
            if (!arr || !arr.length || stride <= 0) return;
            for (let i = stride - 1; i < arr.length; i += stride) {
                arr[i] = 0;
            }
        };
        // 主画布实例
        clear(data.rects, 13);
        clear(data.ringInstances, 9);
        clear(data.imageInstances, 13);
        clear(data.charData, 11);
    }


    window.addEventListener("keydown", e => {

        if (e.ctrlKey && e.key === 'z') {
            if (!undoStack.length) return

            const command = undoStack.pop()
            redoStack.push(command)
            console.log("stacks", undoStack, undoStack.length, redoStack, redoStack.length);

            if (command.type === commandDict.CLICK) {
                let id = command.config
                const select = false
                markSelectedById(id, data.rects, 13, -1, select);
                device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

                markSelectedById(id, data.ringInstances, 9, -1, select);
                device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

                markSelectedById(id, data.imageInstances, 13, -1, select);
                device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

                markSelectedById(id, data.charData, 11, -1, select);
                device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);
            } else if (command.type === commandDict.DRAG) {
                console.log("undo drag");
                const prevData = command.config.before;
                data = prevData;
                device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);
                device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);
                device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);
                device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);
                device.queue.writeBuffer(rectBuffer_mini, 0, data.rects_mini.buffer);
                device.queue.writeBuffer(lineBuffer_mini, 0, data.polylines_mini.buffer);
                device.queue.writeBuffer(arrowInstanceBuffer, 0, data.arrowSegments.buffer);
                device.queue.writeBuffer(lineBuffer, 0, data.polylines.buffer);
            }

        } else if (e.ctrlKey && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) {
            if (!redoStack.length) return
            console.log("ctrlY");

            const command = redoStack.pop()
            undoStack.push(command)
            console.log("stacks", undoStack, undoStack.length, redoStack, redoStack.length);
            if (command.type === commandDict.CLICK) {
                let id = command.config
                const select = true
                markSelectedById(id, data.rects, 13, -1, select);
                device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

                markSelectedById(id, data.ringInstances, 9, -1, select);
                device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

                markSelectedById(id, data.imageInstances, 13, -1, select);
                device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

                markSelectedById(id, data.charData, 11, -1, select);
                device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);
            } else if (command.type === commandDict.DRAG) {
                console.log("redo drag");
                const prevData = command.config.after;
                data = prevData;
                device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);
                device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);
                device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);
                device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);
                device.queue.writeBuffer(rectBuffer_mini, 0, data.rects_mini.buffer);
                device.queue.writeBuffer(lineBuffer_mini, 0, data.polylines_mini.buffer);
                device.queue.writeBuffer(arrowInstanceBuffer, 0, data.arrowSegments.buffer);
                device.queue.writeBuffer(lineBuffer, 0, data.polylines.buffer);
            }

        }
    });
    canvas.addEventListener("mousedown", async e => {
        console.log("down");

        signal.dragging = true; last = [e.clientX, e.clientY];
        const rect = canvas.getBoundingClientRect();
        const px = Math.floor((e.clientX - rect.left));//不要乘以 * devicePixelRatio
        const py = Math.floor((e.clientY - rect.top));
        console.log("pxpy", px, py);

        updateMatrix()
        // 执行 pick 渲染
        await renderPick(device, bindGroup, shaderModule, ringInstanceBuffer, quadBuffer, pickTexture, pipelineLayout, data);

        // 读取 `pickTexture` 中的像素值
        const readBuffer = device.createBuffer({
            size: 4,  // 1像素 RGBA 数据
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture: pickTexture, origin: { x: px, y: py } },
            { buffer: readBuffer, bytesPerRow: 256 },
            [1, 1, 1]
        );
        device.queue.submit([encoder.finish()]);

        // 读取并解析像素数据
        await readBuffer.mapAsync(GPUMapMode.READ);
        const array = new Uint8Array(readBuffer.getMappedRange());
        const [r, g, b, a] = array;
        readBuffer.unmap();

        const id = decodeColorToId(r, g, b);
        console.log("点击选中 ring ID:", r, g, b, id);
        signal.mouseDownID = [r / 255, g / 255, b / 255];
        signal.mouseDownIdFlag = true;


    });
    canvas.addEventListener("mouseup", () => {
        if (undoStack[undoStack.length - 1]?.type === commandDict.DRAG) {
            undoStack[undoStack.length - 1].config.after = cloneData(data);
        }
        signal.dragging = false
        signal.mouseDownIdFlag = false; // 重置鼠标点击 ID
        //  signal.mouseDownID = 0;
        console.log("up");

    });
    canvas.addEventListener("mousemove", e => {
        // console.log("moveid",signal.mouseDownID);
        if (signal.mouseDownIdFlag && !matchColorID(signal.mouseDownID, [0, 0, 0])) {
            if (!undoStack.length || undoStack[undoStack.length - 1]?.type !== commandDict.DRAG) {
                undoStack.push({
                    type: commandDict.DRAG,
                    config: {
                        before: cloneData(data)
                    }
                });
            }
            signal.nodeMoveFlag = true;
            const rect = canvas.getBoundingClientRect();

            const prevX_canvas = (last[0] - rect.left)
            const prevY_canvas = (last[1] - rect.top);

            const currX_canvas = (e.clientX - rect.left)
            const currY_canvas = (e.clientY - rect.top)

            const prevX = prevX_canvas / canvas.width * 2 - 1;
            const prevY = 1 - prevY_canvas / canvas.height * 2;

            const currX = currX_canvas / canvas.width * 2 - 1;
            const currY = 1 - currY_canvas / canvas.height * 2;

            const boundWidth = STYLE.nodeBound.maxX - STYLE.nodeBound.minX;
            const boundHeight = STYLE.nodeBound.maxY - STYLE.nodeBound.minY;

            const scaleToMinimap = Math.max(boundWidth, boundHeight) || 1;
            const miniMoveScale = (1 - 2 * STYLE.minimapMargin) * 2 / scaleToMinimap;

            const [shiftX_canvas, shiftY_canvas] = [currX_canvas - prevX_canvas, currY_canvas - prevY_canvas];

            markMoveById(signal.mouseDownID, data.rects_mini, 13, -1, [shiftX_canvas, shiftY_canvas]);
            device.queue.writeBuffer(rectBuffer_mini, 0, data.rects_mini.buffer);
            edgeMoveById(signal.mouseDownID, data.polylines_mini, 12, -1, [shiftX_canvas, shiftY_canvas]);
            device.queue.writeBuffer(lineBuffer_mini, 0, data.polylines_mini.buffer);

            const inv = mat3.create();
            if (mat3.invert(inv, viewMatrix)) {
                const prevWorldX = inv[0] * prevX + inv[3] * prevY + inv[6];
                const prevWorldY = inv[1] * prevX + inv[4] * prevY + inv[7];

                const currWorldX = inv[0] * currX + inv[3] * currY + inv[6];
                const currWorldY = inv[1] * currX + inv[4] * currY + inv[7];
                const [shiftX, shiftY] = [currWorldX - prevWorldX, currWorldY - prevWorldY];
                markMoveById(signal.mouseDownID, data.rects, 13, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

                markMoveById(signal.mouseDownID, data.ringInstances, 9, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

                markMoveById(signal.mouseDownID, data.imageInstances, 13, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

                markMoveById(signal.mouseDownID, data.charData, 11, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);

                // 先更新边折线，再重算箭头以保持方向一致
                edgeMoveById(signal.mouseDownID, data.polylines, 12, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(lineBuffer, 0, data.polylines.buffer);

                arrowMoveById(signal.mouseDownID, data.arrowSegments, 12, -1, [shiftX, shiftY], data.polylines, 12, -1);
                device.queue.writeBuffer(arrowInstanceBuffer, 0, data.arrowSegments.buffer);




            }

            last = [e.clientX, e.clientY];
        }

        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width * 2 - 1;
        const y = 1 - (e.clientY - rect.top) / canvas.height * 2;
        const inv = mat3.create();
        if (mat3.invert(inv, viewMatrix)) {
            const worldX = inv[0] * x + inv[3] * y + inv[6];
            const worldY = inv[1] * x + inv[4] * y + inv[7];
            // console.log("moveto", worldX, worldY);

            hoverPos1 = [worldX - hoverSize / 2, worldY + hoverSize / 2]; // 左上角
            hoverPos2 = [worldX + hoverSize / 2, worldY - hoverSize / 2]; // 右下角
        } else {
            hoverPos1 = [999, 999];
            hoverPos2 = [999, 999];
        }
        // 如果在拖动，也更新偏移
        if (signal.dragging) {
            signal.canvasMoveFlag = true;
            const rect = canvas.getBoundingClientRect();

            const prevX = (last[0] - rect.left) / canvas.width * 2 - 1;
            const prevY = 1 - (last[1] - rect.top) / canvas.height * 2;

            const currX = (e.clientX - rect.left) / canvas.width * 2 - 1;
            const currY = 1 - (e.clientY - rect.top) / canvas.height * 2;

            const inv = mat3.create();
            if (mat3.invert(inv, viewMatrix)) {
                const prevWorldX = inv[0] * prevX + inv[3] * prevY + inv[6];
                const prevWorldY = inv[1] * prevX + inv[4] * prevY + inv[7];

                const currWorldX = inv[0] * currX + inv[3] * currY + inv[6];
                const currWorldY = inv[1] * currX + inv[4] * currY + inv[7];

                offset[0] += currWorldX - prevWorldX;
                offset[1] += currWorldY - prevWorldY;
            }

            last = [e.clientX, e.clientY];

            device.queue.writeBuffer(viewRectBuffer, 0, getViewRect(canvas, viewMatrix));
        }

        updateMatrix();
    });
    miniMapCanvas.addEventListener("mousedown", async e => {
        console.log("down");

        signal.dragging = true; last = [e.clientX, e.clientY];
        const rect = canvas.getBoundingClientRect();
        const px = Math.floor((e.clientX - rect.left));//不要乘以 * devicePixelRatio
        const py = Math.floor((e.clientY - rect.top));
        console.log("pxpy", px, py);


        signal.mouseDownIdFlag = true;


    });
    miniMapCanvas.addEventListener("mouseup", () => {

        signal.dragging = false
        console.log("up");

    });
    miniMapCanvas.addEventListener("mousemove", e => {

        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width * 2 - 1;
        const y = 1 - (e.clientY - rect.top) / canvas.height * 2;
        const inv = mat3.create();
        if (mat3.invert(inv, viewMatrix)) {
            const worldX = inv[0] * x + inv[3] * y + inv[6];
            const worldY = inv[1] * x + inv[4] * y + inv[7];
            // console.log("moveto", worldX, worldY);

            hoverPos1 = [worldX - hoverSize / 2, worldY + hoverSize / 2]; // 左上角
            hoverPos2 = [worldX + hoverSize / 2, worldY - hoverSize / 2]; // 右下角
        } else {
            hoverPos1 = [999, 999];
            hoverPos2 = [999, 999];
        }
        // 如果在拖动，也更新偏移
        if (signal.dragging) {
            signal.canvasMoveFlag = true;
            const rect = canvas.getBoundingClientRect();

            const prevX = (last[0] - rect.left) / canvas.width * 2 - 1;
            const prevY = 1 - (last[1] - rect.top) / canvas.height * 2;

            const currX = (e.clientX - rect.left) / canvas.width * 2 - 1;
            const currY = 1 - (e.clientY - rect.top) / canvas.height * 2;
            const miniMapPixelScale = (STYLE.nodeBound.maxX - STYLE.nodeBound.minX) / (0.9 * miniMapCanvas.width);
            offset[0] -= (currX - prevX) * miniMapPixelScale;
            offset[1] -= (currY - prevY) * miniMapPixelScale;
            last = [e.clientX, e.clientY];

            device.queue.writeBuffer(viewRectBuffer, 0, getViewRect(canvas, viewMatrix));
        }

        updateMatrix();
    });
    let last = [0, 0];
    updateMatrix();

    updateMiniMapMatrix()


    // === 渲染帧
    function frame() {

        const encoder = device.createCommandEncoder();

        // 🧠 检查是否需要重建 MSAA 纹理（第一次 or 尺寸变了）
        if (!msaaTexture || msaaTexture.width !== canvas.width || msaaTexture.height !== canvas.height) {
            msaaTexture = device.createTexture({
                size: [canvas.width, canvas.height],
                sampleCount,
                format,
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });
        }

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: msaaTexture.createView(),  // 渲染到 MSAA 纹理
                resolveTarget: context.getCurrentTexture().createView(), // 解析结果输出到最终画布
                loadOp: "clear",
                storeOp: "store",
                clearValue: STYLE.clearColor
            }]
        });

        pass.setBindGroup(0, bindGroup);

        pass.setPipeline(linePipeline);
        pass.setVertexBuffer(0, lineBuffer);
        pass.draw(data.polylines.length / 6);

        // pass.setPipeline(rectPipeline);
        // pass.setVertexBuffer(0, quadBuffer);
        // pass.setVertexBuffer(1, rectBuffer);
        // pass.draw(4, data.rects.length / 13);

        pass.setPipeline(imagePipeline);
        pass.setVertexBuffer(0, quadBuffer);
        pass.setVertexBuffer(1, imageInstanceBuffer);
        pass.draw(4, data.imageInstances.length / 13); // 每个 instance 8 个 float

        pass.setPipeline(arrowPipeline);
        pass.setVertexBuffer(0, arrowVertexBuffer);
        pass.setVertexBuffer(1, arrowInstanceBuffer);
        pass.draw(3, data.arrowSegments.length / 12);

        pass.setPipeline(charPipeline);
        pass.setVertexBuffer(0, charQuadBuffer);
        pass.setVertexBuffer(1, charInstanceBuffer);
        pass.draw(4, data.charData.length / 11);

        pass.setPipeline(ringPipeline);
        pass.setVertexBuffer(0, quadBuffer);           // [-0.5, -0.5] ~ [0.5, 0.5]
        pass.setVertexBuffer(1, ringInstanceBuffer);   // 圆环数据
        pass.draw(4, data.ringInstances.length / 9);

        pass.end();
        device.queue.submit([encoder.finish()]);


        // === 小地图渲染逻辑 ===
        const miniMapEncoder = device.createCommandEncoder();
        const miniMapPass = miniMapEncoder.beginRenderPass({
            colorAttachments: [{
                view: miniMapContext.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: { r: 1, g: 1, b: 1, a: 1 }
            }]
        });
        miniMapPass.setBindGroup(0, miniMapBindGroup);

        // 边
        miniMapPass.setPipeline(miniMapLinePipeline);
        miniMapPass.setVertexBuffer(0, lineBuffer_mini);
        miniMapPass.draw(data.polylines_mini.length / 6);

        // 节点矩形
        miniMapPass.setPipeline(miniMapRectPipeline);
        miniMapPass.setVertexBuffer(0, quadBuffer);
        miniMapPass.setVertexBuffer(1, rectBuffer_mini);
        miniMapPass.draw(4, data.rects_mini.length / 13);

        // 绘制主视图框
        miniMapPass.setPipeline(miniMapRectPipeline);
        miniMapPass.setVertexBuffer(0, quadBuffer);
        miniMapPass.setVertexBuffer(1, viewRectBuffer);
        miniMapPass.draw(4, 1);


        miniMapPass.end();
        device.queue.submit([miniMapEncoder.finish()]);

        requestAnimationFrame(frame);
    }


    frame();
}

// === 辅助函数
function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage,
        mappedAtCreation: true
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
}

function createPipeline(device, module, layout, format, vert, frag, buffers, topology, sampleCount) {
    return device.createRenderPipeline({
        layout,
        vertex: { module, entryPoint: vert, buffers },
        fragment: {
            module,
            entryPoint: frag,
            targets: [{
                format,
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'zero',
                        operation: 'add'
                    }
                }
            }]
        },
        primitive: { topology },
        multisample: { count: sampleCount } // ✅ 添加此项
    });
}

function collectCharsFromGraph(graph) {
    const charSet = new Set();
    graph.nodes.forEach(node => {
        const idStr = node.label.toString();
        for (const ch of idStr) {
            charSet.add(ch);
        }
    });
    return Array.from(charSet).sort(); // 排序确保一致性
}


function generateDigitTextureCanvas() {
    const canvas = document.createElement("canvas");
    const cellW = 32, cellH = 32;
    canvas.width = cellW * 10;
    canvas.height = cellH;

    const ctx = canvas.getContext("2d", { alpha: true });

    // ✅ 设置黄底
    // ctx.fillStyle = "#FFFF00"; // bright yellow
    // ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ✅ 设置黑字
    ctx.fillStyle = "black";
    ctx.font = "24px sans-serif";//字在图集中的大小越大，图集的字就越清晰，占的纹理空间也越多（可能会让单个字符图占据更大的纹理区域，UV计算不变）
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < 10; i++) {
        const x = i * cellW + cellW / 2;
        const y = cellH / 2;
        ctx.fillText(i.toString(), x, y);
    }

    // ✅ 调试可见性
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    console.log("Top-left pixel RGBA:", imageData.data.slice(0, 4)); // [R, G, B, A]
    document.body.appendChild(canvas);

    return canvas;
}

function generateCharTextureCanvas(charList) {
    const fontSize = 72;
    const padding = 40;
    const maxTextureWidth = 8192;

    const ctx = document.createElement("canvas").getContext("2d");
    ctx.font = `${fontSize}px sans-serif`;

    const charWidths = charList.map(ch => ctx.measureText(ch).width + 2 * padding);
    const cellHeights = fontSize * 1.8;

    // 累积宽度，自动换行
    const rows = [];
    let currentRow = [];
    let currentRowWidth = 0;

    for (let i = 0; i < charList.length; i++) {
        const w = charWidths[i];
        if (currentRowWidth + w > maxTextureWidth) {
            rows.push(currentRow);
            currentRow = [];
            currentRowWidth = 0;
        }
        currentRow.push({ char: charList[i], width: w });
        currentRowWidth += w;
    }
    if (currentRow.length > 0) rows.push(currentRow);

    const canvasWidth = maxTextureWidth;
    const canvasHeight = rows.length * cellHeights;

    const canvas = document.createElement("canvas");
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const ctx2 = canvas.getContext("2d", { alpha: true });
    ctx2.clearRect(0, 0, canvas.width, canvas.height);
    ctx2.fillStyle = "black";
    ctx2.font = `${fontSize}px sans-serif`;
    ctx2.textAlign = "left";
    ctx2.textBaseline = "middle";

    const uvMap = {};
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        let x = 0;
        const row = rows[rowIndex];
        for (const item of row) {
            const y = rowIndex * cellHeights + cellHeights / 2;
            ctx2.fillText(item.char, x + padding, y);

            const epsU = 0.5 / canvas.width;
            const epsV = 0.5 / canvas.height;
            const u0 = x / canvas.width + epsU;
            const u1 = (x + item.width) / canvas.width - epsU;
            const v0 = rowIndex * cellHeights / canvas.height + epsV;
            const v1 = (rowIndex * cellHeights + cellHeights) / canvas.height - epsV;

            uvMap[item.char] = [u0, u1, v0, v1];
            x += item.width;
        }
    }

    return { canvas, uvMap };
}

async function generateImageAtlas(imageList) {
    const imgSize = 256;
    const rowLen = Math.ceil(Math.sqrt(imageList.length));

    const canvas = document.createElement("canvas");
    canvas.width = rowLen * imgSize;
    canvas.height = rowLen * imgSize;

    const ctx = canvas.getContext("2d", { alpha: true });
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const uvMap = {};

    // 异步加载所有图片
    const loadImage = src => new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve({ src, img });
        img.src = src;
    });

    const loadedImages = await Promise.all(imageList.map(loadImage));

    // 绘制图集并记录 UV
    loadedImages.forEach(({ src, img }, idx) => {
        const row = Math.floor(idx / rowLen);
        const col = idx % rowLen;
        const x = col * imgSize;
        const y = row * imgSize;
        ctx.drawImage(img, x, y, imgSize, imgSize);
        uvMap[src] = [
            x / canvas.width, (x + imgSize) / canvas.width,
            y / canvas.height, (y + imgSize) / canvas.height
        ];
    });
    return { canvas, uvMap };
}


async function generateDigitTexture(device, charList) {
    const { canvas, uvMap } = generateCharTextureCanvas(charList);
    const levelCount = Math.floor(Math.log2(Math.max(canvas.width, canvas.height))) + 1;
    const texture = uploadTextureByImageData(device, canvas, { enableSRGB: true, generateMipmaps: true });
    const sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
        lodMinClamp: 0,
        lodMaxClamp: levelCount - 1
    });
    const samplerNearest = device.createSampler({
        magFilter: "nearest",
        minFilter: "nearest",
        mipmapFilter: "nearest",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
        lodMinClamp: 0,
        lodMaxClamp: levelCount - 1
    });
    return { texture, samplerLinear: sampler, samplerNearest, uvMap };
}

function uploadTextureByImageData(device, sourceCanvas, options = {}) {
    const { enableSRGB = false, generateMipmaps = false } = options;
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const levelCount = generateMipmaps ? (Math.floor(Math.log2(Math.max(width, height))) + 1) : 1;
    const format = enableSRGB ? "rgba8unorm-srgb" : "rgba8unorm";

    const texture = device.createTexture({
        size: [width, height],
        format,
        mipLevelCount: levelCount,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    let canvas = sourceCanvas;
    let w = width;
    let h = height;

    for (let level = 0; level < levelCount; level++) {
        const ctx = canvas.getContext("2d");
        const imageData = ctx.getImageData(0, 0, w, h);
        const rgba = new Uint8Array(imageData.data);

        const bytesPerPixel = 4;
        const unpaddedBytesPerRow = w * bytesPerPixel;
        const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
        const paddedData = new Uint8Array(paddedBytesPerRow * h);

        for (let y = 0; y < h; y++) {
            const srcOffset = y * unpaddedBytesPerRow;
            const dstOffset = y * paddedBytesPerRow;
            paddedData.set(rgba.subarray(srcOffset, srcOffset + unpaddedBytesPerRow), dstOffset);
        }

        device.queue.writeTexture(
            { texture, mipLevel: level, origin: { x: 0, y: 0 } },
            paddedData,
            { bytesPerRow: paddedBytesPerRow, rowsPerImage: h },
            { width: w, height: h }
        );

        if (generateMipmaps && level + 1 < levelCount) {
            const nextW = Math.max(1, Math.floor(w / 2));
            const nextH = Math.max(1, Math.floor(h / 2));
            const downCanvas = document.createElement("canvas");
            downCanvas.width = nextW;
            downCanvas.height = nextH;
            const downCtx = downCanvas.getContext("2d", { alpha: true });
            downCtx.imageSmoothingEnabled = true;
            downCtx.imageSmoothingQuality = "high";
            downCtx.drawImage(canvas, 0, 0, w, h, 0, 0, nextW, nextH);
            canvas = downCanvas;
            w = nextW;
            h = nextH;
        }
    }

    return texture;
}

function createCharUVMap(charList) {
    const uvMap = {};
    const cellWidth = 1 / charList.length;

    charList.forEach((ch, i) => {
        uvMap[ch] = [i * cellWidth, (i + 1) * cellWidth];
    });

    return uvMap;
}


function extractDataFromG6(graph, canvas, uvMap, imageUVMap) {
    function scale(px, py) {
        return [(px / canvas.width) * 2 - 1, 1 - (py / canvas.height) * 2];
    }

    let rects = [],
        polylines = [],
        arrows = [],
        chars = [],
        imageInstances = [],
        ringInstances = [],
        rects_mini = [],
        polylines_mini = [];
    // 偏移量（贴图相对矩形中心位置）
    const imageOffset = -0.0;
    const strokeWidth = 0.2; // 20% 的宽度

    // const bounds = getGraphBounds(graph.nodes);  // 获取坐标范围
    // const a = getEquivalentNDCMatrix(bounds); console.log("1212", a);

    // const toNDCWithSize = createNDCMapperWithSize(bounds);
    // const toNDC = createNDCMapper(bounds);
    STYLE.nodeBound = getGraphBounds(graph.nodes);
    // let DataManager.nodeColorIdMap = new Map();
    graph.nodes.forEach((node, index) => {
        const nodeID = encodeIdToColor(index + 1); console.log(nodeID);
        DataManager.nodeColorIdMap.set(node.id, nodeID);
        DataManager.adjGraph.set([nodeID[0], nodeID[1], nodeID[2]].join(","), new Set())
        DataManager.adjGraph_arrow.set([nodeID[0], nodeID[1], nodeID[2]].join(","), new Set())
        const selected = 0;
        const [x, y] = scale(node.x, node.y);
        const w = node.size / canvas.width * 2;//for JinAn data
        const h = node.size / canvas.height * 2;
        // const w = node.size[0] / canvas.width * 2;//for G6
        // const h = node.size[1] / canvas.height * 2;
        // const w = node.width / canvas.width * 2;//for GPU
        // const h = node.height / canvas.height * 2;
        rects.push(x, y, w, h, ...STYLE.nodeColor, ...nodeID, selected);
        // const [x_mini, y_mini, w_mini, h_mini] = toNDCWithSize(node.x, node.y, node.size);
        // rects_mini.push(x_mini, y_mini, w_mini, h_mini, ...STYLE.nodeColor, ...nodeID, selected);
        rects_mini.push(node.x, node.y, node.size, node.size, ...STYLE.nodeColor_mini, ...nodeID, selected);

        const idStr = node.label.toString();
        const charSize = STYLE.charSize;
        const charWidthRatio = STYLE.charWidthRatio || 1.0;
        const charSpacing = charSize * charWidthRatio * STYLE.charSpacingRatio;
        const step = charSize * charWidthRatio + charSpacing;
        const baseX = x - (idStr.length - 1) * step * 0.5;//字符相对节点中心位置的偏移

        for (let i = 0; i < idStr.length; i++) {
            const ch = idStr[i];
            const [u0, u1, v0, v1] = uvMap[ch] || [0, 0.1, 0, 1];
            chars.push(baseX + i * step, y + STYLE.charShiftY, u0, u1, v0, v1, ...nodeID, selected);
        }


        const uv = imageUVMap["../data/1.png"] || [0, 0.1, 0, 0.1];
        imageInstances.push(x + imageOffset, y + imageOffset, w / 2, h / 2, ...uv, ...nodeID, selected)

        const radius = w * 1.25;
        ringInstances.push(x, y, radius, strokeWidth, ...nodeID, selected)
    });
    console.log(DataManager.adjGraph);
    let edgeOffsetCount = 0;
    let arrowOffsetCount = 0;
    graph.edges.forEach((edge) => {
        edge.sourceColorId = DataManager.nodeColorIdMap.get(edge.source) || encodeIdToColor(0);
        edge.targetColorId = DataManager.nodeColorIdMap.get(edge.target) || encodeIdToColor(0);
        if (edge.source == edge.target) {
            const direction = edge.loopCfg.position;
            const loopoffsetX = edge.loopCfg.dist / 4
            const loopoffsetY = edge.loopCfg.dist / 2
            // const [loopCenterX,loopCenterY] = [(edge.startPoint.x + edge.endPoint.x)/2, (edge.startPoint.y + edge.endPoint.y)/2] 
            const [sourceX, sourceY] = scale(edge.startPoint.x, edge.startPoint.y)
            const [mid1_X, mid1_Y] = scale(edge.startPoint.x - loopoffsetX, edge.startPoint.y - loopoffsetY)
            const [mid2_X, mid2_Y] = scale(edge.endPoint.x + loopoffsetX, edge.endPoint.y - loopoffsetY)
            const [targetX, targetY] = scale(edge.endPoint.x, edge.endPoint.y)

            polylines.push(sourceX, sourceY, ...edge.sourceColorId, mid1_X, mid1_Y, ...edge.targetColorId);
            polylines_mini.push(edge.startPoint.x, edge.startPoint.y, ...edge.sourceColorId, edge.startPoint.x - loopoffsetX, edge.startPoint.y - loopoffsetY, ...edge.targetColorId);
            DataManager.adjGraph.get([edge.sourceColorId[0], edge.sourceColorId[1], edge.sourceColorId[2]].join(",")).add(edgeOffsetCount);
            DataManager.adjGraph.get([edge.targetColorId[0], edge.targetColorId[1], edge.targetColorId[2]].join(",")).add(edgeOffsetCount++);

            polylines.push(mid1_X, mid1_Y, ...edge.sourceColorId, mid2_X, mid2_Y, ...edge.targetColorId);
            polylines_mini.push(edge.startPoint.x - loopoffsetX, edge.startPoint.y - loopoffsetY, ...edge.sourceColorId, edge.endPoint.x + loopoffsetX, edge.endPoint.y - loopoffsetY, ...edge.targetColorId);
            DataManager.adjGraph.get([edge.sourceColorId[0], edge.sourceColorId[1], edge.sourceColorId[2]].join(",")).add(edgeOffsetCount);
            DataManager.adjGraph.get([edge.targetColorId[0], edge.targetColorId[1], edge.targetColorId[2]].join(",")).add(edgeOffsetCount++);

            polylines.push(mid2_X, mid2_Y, ...edge.sourceColorId, targetX, targetY, ...edge.targetColorId);
            polylines_mini.push(edge.endPoint.x + loopoffsetX, edge.endPoint.y - loopoffsetY, ...edge.sourceColorId, edge.endPoint.x, edge.endPoint.y, ...edge.targetColorId);
            DataManager.adjGraph.get([edge.sourceColorId[0], edge.sourceColorId[1], edge.sourceColorId[2]].join(",")).add(edgeOffsetCount);
            DataManager.adjGraph.get([edge.targetColorId[0], edge.targetColorId[1], edge.targetColorId[2]].join(",")).add(edgeOffsetCount++);
            arrows.push(mid2_X, mid2_Y, targetX, targetY, ...edge.sourceColorId, ...edge.targetColorId);
            DataManager.adjGraph_arrow.get([edge.sourceColorId[0], edge.sourceColorId[1], edge.sourceColorId[2]].join(",")).add(arrowOffsetCount);
            DataManager.adjGraph_arrow.get([edge.targetColorId[0], edge.targetColorId[1], edge.targetColorId[2]].join(",")).add(arrowOffsetCount++);
        } else {
            const [x1, y1] = scale(edge.startPoint.x, edge.startPoint.y);
            const [x2, y2] = scale(edge.endPoint.x, edge.endPoint.y);
            // 使用起点终点构成的矩形角点，构造“中心对称”的四次贝塞尔：P1=(sx,ty), P2=中心, P3=(tx,sy)
            const sx = x1, sy = y1;
            const tx = x2, ty = y2;
            const midx = (sx + tx) * 0.5;
            const midy = (sy + ty) * 0.5;
            const alphaX = (STYLE.edgeCurveOffsetRatioX ?? 0.25); // X方向插值比例
            const alphaY = (STYLE.edgeCurveOffsetRatioY ?? 0.25); // Y方向插值比例
            const p1x = tx * (1 - alphaX) + midx * alphaX; // 终点角点 -> 中心，X方向插值
            const p1y = sy * (1 - alphaY) + midy * alphaY; // 终点角点 -> 中心，Y方向插值
            const p2x = midx, p2y = midy; // 中心点，保证中心对称
            const p3x = sx * (1 - alphaX) + midx * alphaX; // 起点角点 -> 中心，X方向插值
            const p3y = ty * (1 - alphaY) + midy * alphaY; // 起点角点 -> 中心，Y方向插值

            const segs = Math.max(8, Math.floor(STYLE.edgeCurveSegments ?? 24));
            let ax0_last = sx, ay0_last = sy, ax1_last = sx, ay1_last = sy; // 记录最后一段采样
            for (let i = 0; i < segs; i++) {
                const t0 = i / segs;
                const t1 = (i + 1) / segs;
                const u0 = 1 - t0, u1 = 1 - t1;
                const ax0 = u0*u0*u0*u0 * sx + 4*u0*u0*u0*t0 * p1x + 6*u0*u0*t0*t0 * p2x + 4*u0*t0*t0*t0 * p3x + t0*t0*t0*t0 * tx;
                const ay0 = u0*u0*u0*u0 * sy + 4*u0*u0*u0*t0 * p1y + 6*u0*u0*t0*t0 * p2y + 4*u0*t0*t0*t0 * p3y + t0*t0*t0*t0 * ty;
                const ax1 = u1*u1*u1*u1 * sx + 4*u1*u1*u1*t1 * p1x + 6*u1*u1*t1*t1 * p2x + 4*u1*t1*t1*t1 * p3x + t1*t1*t1*t1 * tx;
                const ay1 = u1*u1*u1*u1 * sy + 4*u1*u1*u1*t1 * p1y + 6*u1*u1*t1*t1 * p2y + 4*u1*t1*t1*t1 * p3y + t1*t1*t1*t1 * ty;
                polylines.push(ax0, ay0, ...edge.sourceColorId, ax1, ay1, ...edge.targetColorId);
                ax0_last = ax0; ay0_last = ay0; ax1_last = ax1; ay1_last = ay1; // 更新最后一段
                // 最小地图：用像素坐标采样同样的“四次中心对称曲线”
                const csx = edge.startPoint.x, csy = edge.startPoint.y;
                const ctx = edge.endPoint.x, cty = edge.endPoint.y;
                const cmidx = (csx + ctx) * 0.5;
                const cmidy = (csy + cty) * 0.5;
                const cp1x = ctx * (1 - alphaX) + cmidx * alphaX; // 终点角点 -> 中心，X方向插值
                const cp1y = csy * (1 - alphaY) + cmidy * alphaY; // 终点角点 -> 中心，Y方向插值
                const cp2x = cmidx, cp2y = cmidy;
                const cp3x = csx * (1 - alphaX) + cmidx * alphaX; // 起点角点 -> 中心，X方向插值
                const cp3y = cty * (1 - alphaY) + cmidy * alphaY; // 起点角点 -> 中心，Y方向插值
                const cx0 = (1 - t0)**4 * csx + 4*(1 - t0)**3 * t0 * cp1x + 6*(1 - t0)**2 * t0**2 * cp2x + 4*(1 - t0) * t0**3 * cp3x + t0**4 * ctx;
                const cy0 = (1 - t0)**4 * csy + 4*(1 - t0)**3 * t0 * cp1y + 6*(1 - t0)**2 * t0**2 * cp2y + 4*(1 - t0) * t0**3 * cp3y + t0**4 * cty;
                const cx1 = (1 - t1)**4 * csx + 4*(1 - t1)**3 * t1 * cp1x + 6*(1 - t1)**2 * t1**2 * cp2x + 4*(1 - t1) * t1**3 * cp3x + t1**4 * ctx;
                const cy1 = (1 - t1)**4 * csy + 4*(1 - t1)**3 * t1 * cp1y + 6*(1 - t1)**2 * t1**2 * cp2y + 4*(1 - t1) * t1**3 * cp3y + t1**4 * cty;
                polylines_mini.push(cx0, cy0, ...edge.sourceColorId, cx1, cy1, ...edge.targetColorId);

                DataManager.adjGraph.get([edge.sourceColorId[0], edge.sourceColorId[1], edge.sourceColorId[2]].join(",")).add(edgeOffsetCount);
                DataManager.adjGraph.get([edge.targetColorId[0], edge.targetColorId[1], edge.targetColorId[2]].join(",")).add(edgeOffsetCount++);
            }

            // 箭头使用四次贝塞尔在 t=1 的切线方向：B'(1) = 4*(P4 - P1)（靠近终点的角点）
            const tx_dx = 4 * (tx - p1x);
            const tx_dy = 4 * (ty - p1y);
            const L_last = Math.hypot(ax1_last - ax0_last, ay1_last - ay0_last) || 1.0;
            const nearDist = (STYLE.arrowSize ?? 0.03) * 0.5;
            const r = Math.min(1.0, nearDist / L_last);
            const px1 = x2 * (1 - r) + ax0_last * r; // 基点位于最后一段折线上
            const py1 = y2 * (1 - r) + ay0_last * r;
            arrows.push(px1, py1, x2, y2, ...edge.sourceColorId, ...edge.targetColorId);
            DataManager.adjGraph_arrow.get([edge.sourceColorId[0], edge.sourceColorId[1], edge.sourceColorId[2]].join(",")).add(arrowOffsetCount);
            DataManager.adjGraph_arrow.get([edge.targetColorId[0], edge.targetColorId[1], edge.targetColorId[2]].join(",")).add(arrowOffsetCount++);
        }

    });
    return {
        rects: new Float32Array(rects),
        polylines: new Float32Array(polylines),
        arrowSegments: new Float32Array(arrows),
        charData: new Float32Array(chars),
        imageInstances: new Float32Array(imageInstances),
        ringInstances: new Float32Array(ringInstances),

        rects_mini: new Float32Array(rects_mini),
        polylines_mini: new Float32Array(polylines_mini)
    };
}

function encodeIdToColor(id) {
    return [
        (id & 0xff) / 255,
        ((id >> 8) & 0xff) / 255,
        ((id >> 16) & 0xff) / 255,
        1.0 // ✅ 强制 alpha 非 0
    ];
}

function decodeColorToId(r, g, b) {
    return r + (g << 8) + (b << 16); // ✅ 只使用 RGB
}
function decodeColorToIdFloat(r, g, b) {
    return Math.round(r * 255) + (Math.round(g * 255) << 8) + (Math.round(b * 255) << 16);
}

async function renderPick(device, bindGroup, shaderModule, ringInstanceBuffer, quadBuffer, pickTexture, pipelineLayout, data) {
    // 创建 pick pipeline
    const ringPickPipeline = createPipeline(
        device,
        shaderModule,
        pipelineLayout,
        "rgba8unorm",
        "ring_pick_vertex", // 使用你定义的 vertex shader
        "ring_pick_frag",   // 使用输出 pickColor 的 fragment shader
        [
            { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
            {
                arrayStride: 36, stepMode: "instance", attributes: [
                    { shaderLocation: 1, format: "float32x2", offset: 0 },    // center
                    { shaderLocation: 2, format: "float32", offset: 8 },      // radius
                    { shaderLocation: 3, format: "float32", offset: 12 },     // stroke width
                    { shaderLocation: 4, format: "float32x4", offset: 16 },   // pickColor
                    { shaderLocation: 5, format: "float32", offset: 32 }
                ]
            }
        ],
        "triangle-strip", 1
    );

    // 创建 command encoder 和 render pass
    const encoder = device.createCommandEncoder();

    const pickPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: pickTexture.createView(),  // 渲染到 pickTexture
            loadOp: "clear",                 // 清空
            storeOp: "store",                // 保存渲染结果
            clearValue: { r: 0, g: 0, b: 0, a: 0 },  // 预设为黑色（未选中）
            sampleCount: 1              // 禁用 MSAA
        }]
    });

    // 设置 bindGroup 和 pipeline
    pickPass.setBindGroup(0, bindGroup);
    pickPass.setPipeline(ringPickPipeline);
    pickPass.setVertexBuffer(0, quadBuffer);
    pickPass.setVertexBuffer(1, ringInstanceBuffer);  // 设置 ring 实例数据
    pickPass.draw(4, data.ringInstances.length / 9);  // 渲染所有的 ring
    pickPass.end();

    // 提交命令
    device.queue.submit([encoder.finish()]);
}

//根据ID在data的子数据中找到对应的索引（注意是随机查找，如果里面元素的id不从1开始需要做映射
// function markSelectedById(id, arr, stride, selectedOffset) {
//     arr[id * stride + selectedOffset] = 1
// }

//最后四位是颜色编码的ID
function markSelectedById(colorId, arr, stride, selectedOffset, select = true) {
    for (let i = 1; i * stride <= arr.length; i++) {
        let [r, g, b] = [arr[i * stride + selectedOffset - 4], arr[i * stride + selectedOffset - 3], arr[i * stride + selectedOffset - 2]]
        if (matchColorID([r, g, b], colorId)) {
            if (select) {
                arr[i * stride + selectedOffset] = 1

            } else {
                arr[i * stride + selectedOffset] = 0
            }

        }
    }
}

//第一、二位是位置
function markMoveById(id, arr, stride, selectedOffset, [shiftX, shiftY]) {
    for (let i = 1; i * stride <= arr.length; i++) {
        let [r, g, b] = [arr[i * stride + selectedOffset - 4], arr[i * stride + selectedOffset - 3], arr[i * stride + selectedOffset - 2]]

        if (matchColorID([r, g, b], id)) {
            arr[(i - 1) * stride] += shiftX
            arr[(i - 1) * stride + 1] += shiftY
        }
    }
}

function arrowMoveById(id, arr, stride, selectedOffset, [shiftX, shiftY], polyArr = null, polyStride = 12, polySelectedOffset = -1) {
    const strColorId = id.join(",");
    const movedId = decodeColorToIdFloat(...id);
    const arrowIdxSet = DataManager.adjGraph_arrow.get(strColorId) ?? new Set();

    for (const value of arrowIdxSet) {
        const i = value + 1; // 数据从 1 开始
        const rS = arr[i * stride + selectedOffset - 7];
        const gS = arr[i * stride + selectedOffset - 6];
        const bS = arr[i * stride + selectedOffset - 5];
        const rT = arr[i * stride + selectedOffset - 3];
        const gT = arr[i * stride + selectedOffset - 2];
        const bT = arr[i * stride + selectedOffset - 1];
        const sId = decodeColorToIdFloat(rS, gS, bS);
        const tId = decodeColorToIdFloat(rT, gT, bT);

        // 自环：直接整体平移箭头，保持原有形状与方向
        if (sId === tId && sId === movedId) {
            arr[(i - 1) * stride + 0] += shiftX;
            arr[(i - 1) * stride + 1] += shiftY;
            arr[(i - 1) * stride + 2] += shiftX;
            arr[(i - 1) * stride + 3] += shiftY;
            continue;
        }

        if (!polyArr) {
            // 回退：若无折线数据，则按旧逻辑平移箭头端点
            if (sId === movedId) {
                arr[(i - 1) * stride + 0] += shiftX;
                arr[(i - 1) * stride + 1] += shiftY;
            }
            if (tId === movedId) {
                arr[(i - 1) * stride + 2] += shiftX;
                arr[(i - 1) * stride + 3] += shiftY;
            }
            continue;
        }

        // 通过 sourceId 在邻接集中筛选出与 targetId 匹配的所有分段索引
        const sourceStr = [rS, gS, bS].join(",");
        const targetStr = [rT, gT, bT].join(",");
        const segIdxSet = DataManager.adjGraph.get(sourceStr) ?? new Set();
        const segIdxList = [];
        for (const segValue of segIdxSet) {
            const si = segValue + 1;
            const rTargetSeg = polyArr[si * polyStride + polySelectedOffset - 3];
            const gTargetSeg = polyArr[si * polyStride + polySelectedOffset - 2];
            const bTargetSeg = polyArr[si * polyStride + polySelectedOffset - 1];
            if (rTargetSeg === rT && gTargetSeg === gT && bTargetSeg === bT) {
                segIdxList.push(segValue);
            }
        }
        if (segIdxList.length === 0) {
            // 若未找到分段，保持旧逻辑平移
            if (sId === movedId) {
                arr[(i - 1) * stride + 0] += shiftX;
                arr[(i - 1) * stride + 1] += shiftY;
            }
            if (tId === movedId) {
                arr[(i - 1) * stride + 2] += shiftX;
                arr[(i - 1) * stride + 3] += shiftY;
            }
            continue;
        }
        segIdxList.sort((a, b) => a - b);
        const lastSegI = segIdxList[segIdxList.length - 1] + 1;
        const ax0 = polyArr[(lastSegI - 1) * polyStride + 0];
        const ay0 = polyArr[(lastSegI - 1) * polyStride + 1];
        const ax1 = polyArr[(lastSegI - 1) * polyStride + 6];
        const ay1 = polyArr[(lastSegI - 1) * polyStride + 7];

        // 以最后一段为方向，箭头尖端贴合终点，基点按 nearDist 后退
        const L = Math.hypot(ax1 - ax0, ay1 - ay0) || 1.0;
        const nearDist = (STYLE.arrowSize ?? 0.03) * 0.5;
        const r = Math.min(1.0, nearDist / L);
        const px1 = ax1 * (1 - r) + ax0 * r;
        const py1 = ay1 * (1 - r) + ay0 * r;
        const tipX = ax1;
        const tipY = ay1;

        // 写回箭头实例
        arr[(i - 1) * stride + 0] = px1;
        arr[(i - 1) * stride + 1] = py1;
        arr[(i - 1) * stride + 2] = tipX;
        arr[(i - 1) * stride + 3] = tipY;
    }
}

function edgeMoveById(id, arr, stride, selectedOffset, [shiftX, shiftY]) {
    const strColorId = id.join(",");
    const movedId = decodeColorToIdFloat(...id);
    const indices = Array.from(DataManager.adjGraph.get(strColorId) ?? []);
    if (indices.length === 0) return;

    // 收集自环分段：source 与 target 相同且等于被拖动的节点
    const selfLoopIdx = [];

    // 将属于同一条边的分段按“另一端节点ID”分组（仅非自环）
    const groups = new Map(); // otherIdStr -> {idxList: number[], sourceIsMoved: boolean}
    for (const value of indices) {
        const i = value + 1; // 数据从1开始
        const rS = arr[i * stride + selectedOffset - 9];
        const gS = arr[i * stride + selectedOffset - 8];
        const bS = arr[i * stride + selectedOffset - 7];
        const rT = arr[i * stride + selectedOffset - 3];
        const gT = arr[i * stride + selectedOffset - 2];
        const bT = arr[i * stride + selectedOffset - 1];
        const sId = decodeColorToIdFloat(rS, gS, bS);
        const tId = decodeColorToIdFloat(rT, gT, bT);

        // 自环分段直接记录，后续整体平移
        if (sId === tId && sId === movedId) {
            selfLoopIdx.push(value);
            continue;
        }

        const sourceIsMoved = (sId === movedId);
        const otherStr = [sourceIsMoved ? rT : rS, sourceIsMoved ? gT : gS, sourceIsMoved ? bT : bS].join(",");
        const g = groups.get(otherStr) ?? { idxList: [], sourceIsMoved };
        g.idxList.push(value);
        g.sourceIsMoved = sourceIsMoved; // 保持一致
        groups.set(otherStr, g);
    }

    // 先处理自环：三段线段整体平移，不做贝塞尔重采样
    for (const value of selfLoopIdx) {
        const i = value + 1;
        arr[(i - 1) * stride + 0] += shiftX;
        arr[(i - 1) * stride + 1] += shiftY;
        arr[(i - 1) * stride + 6] += shiftX;
        arr[(i - 1) * stride + 7] += shiftY;
    }

    const alphaX = (STYLE.edgeCurveOffsetRatioX ?? 0.25);
    const alphaY = (STYLE.edgeCurveOffsetRatioY ?? 0.25);

    // 对每一条非自环边整段重算采样
    for (const [_, g] of groups.entries()) {
        const idxList = g.idxList.sort((a, b) => a - b);
        const segCount = idxList.length;
        if (segCount === 0) continue;
        const startI = idxList[0] + 1;
        const endI = idxList[segCount - 1] + 1;
        // 读取当前端点坐标
        let sx = arr[(startI - 1) * stride + 0];
        let sy = arr[(startI - 1) * stride + 1];
        let tx = arr[(endI - 1) * stride + 6];
        let ty = arr[(endI - 1) * stride + 7];
        // 应用拖动位移到移动的端点
        if (g.sourceIsMoved) {
            sx += shiftX; sy += shiftY;
        } else {
            tx += shiftX; ty += shiftY;
        }
        const midx = (sx + tx) * 0.5;
        const midy = (sy + ty) * 0.5;
        const p1x = tx * (1 - alphaX) + midx * alphaX;
        const p1y = sy * (1 - alphaY) + midy * alphaY;
        const p2x = midx, p2y = midy;
        const p3x = sx * (1 - alphaX) + midx * alphaX;
        const p3y = ty * (1 - alphaY) + midy * alphaY;

        for (let j = 0; j < segCount; j++) {
            const value = idxList[j];
            const i = value + 1;
            const t0 = j / segCount;
            const t1 = (j + 1) / segCount;
            const u0 = 1 - t0, u1 = 1 - t1;
            const ax0 = u0*u0*u0*u0 * sx + 4*u0*u0*u0*t0 * p1x + 6*u0*u0*t0*t0 * p2x + 4*u0*t0*t0*t0 * p3x + t0*t0*t0*t0 * tx;
            const ay0 = u0*u0*u0*u0 * sy + 4*u0*u0*u0*t0 * p1y + 6*u0*u0*t0*t0 * p2y + 4*u0*t0*t0*t0 * p3y + t0*t0*t0*t0 * ty;
            const ax1 = u1*u1*u1*u1 * sx + 4*u1*u1*u1*t1 * p1x + 6*u1*u1*t1*t1 * p2x + 4*u1*t1*t1*t1 * p3x + t1*t1*t1*t1 * tx;
            const ay1 = u1*u1*u1*u1 * sy + 4*u1*u1*u1*t1 * p1y + 6*u1*u1*t1*t1 * p2y + 4*u1*t1*t1*t1 * p3y + t1*t1*t1*t1 * ty;
            // 写回当前分段坐标
            arr[(i - 1) * stride + 0] = ax0;
            arr[(i - 1) * stride + 1] = ay0;
            arr[(i - 1) * stride + 6] = ax1;
            arr[(i - 1) * stride + 7] = ay1;
        }
    }
}

function mini_EdgeMoveById(id, arr, stride, selectedOffset, [shiftX, shiftY]) {
    const strColorId = id.join(",");
    const movedId = decodeColorToIdFloat(...id);
    const indices = Array.from(DataManager.adjGraph.get(strColorId) ?? []);
    if (indices.length === 0) return;

    const groups = new Map(); // otherIdStr -> {idxList: number[], sourceIsMoved: boolean}
    for (const value of indices) {
        const i = value + 1;
        const rS = arr[i * stride + selectedOffset - 9];
        const gS = arr[i * stride + selectedOffset - 8];
        const bS = arr[i * stride + selectedOffset - 7];
        const rT = arr[i * stride + selectedOffset - 3];
        const gT = arr[i * stride + selectedOffset - 2];
        const bT = arr[i * stride + selectedOffset - 1];
        const sId = decodeColorToIdFloat(rS, gS, bS);
        const tId = decodeColorToIdFloat(rT, gT, bT);
        const sourceIsMoved = (sId === movedId);
        const otherStr = [sourceIsMoved ? rT : rS, sourceIsMoved ? gT : gS, sourceIsMoved ? bT : bS].join(",");
        const g = groups.get(otherStr) ?? { idxList: [], sourceIsMoved };
        g.idxList.push(value);
        g.sourceIsMoved = sourceIsMoved;
        groups.set(otherStr, g);
    }

    const alphaX = (STYLE.edgeCurveOffsetRatioX ?? 0.25);
    const alphaY = (STYLE.edgeCurveOffsetRatioY ?? 0.25);

    for (const [_, g] of groups.entries()) {
        const idxList = g.idxList.sort((a, b) => a - b);
        const segCount = idxList.length;
        if (segCount === 0) continue;
        const startI = idxList[0] + 1;
        const endI = idxList[segCount - 1] + 1;
        let csx = arr[(startI - 1) * stride + 0];
        let csy = arr[(startI - 1) * stride + 1];
        let ctx = arr[(endI - 1) * stride + 6];
        let cty = arr[(endI - 1) * stride + 7];
        if (g.sourceIsMoved) { csx += shiftX; csy += shiftY; } else { ctx += shiftX; cty += shiftY; }
        const cmidx = (csx + ctx) * 0.5;
        const cmidy = (csy + cty) * 0.5;
        const cp1x = ctx * (1 - alphaX) + cmidx * alphaX;
        const cp1y = csy * (1 - alphaY) + cmidy * alphaY;
        const cp2x = cmidx, cp2y = cmidy;
        const cp3x = csx * (1 - alphaX) + cmidx * alphaX;
        const cp3y = cty * (1 - alphaY) + cmidy * alphaY;

        for (let j = 0; j < segCount; j++) {
            const value = idxList[j];
            const i = value + 1;
            const t0 = j / segCount;
            const t1 = (j + 1) / segCount;
            const cx0 = (1 - t0)**4 * csx + 4*(1 - t0)**3 * t0 * cp1x + 6*(1 - t0)**2 * t0**2 * cp2x + 4*(1 - t0) * t0**3 * cp3x + t0**4 * ctx;
            const cy0 = (1 - t0)**4 * csy + 4*(1 - t0)**3 * t0 * cp1y + 6*(1 - t0)**2 * t0**2 * cp2y + 4*(1 - t0) * t0**3 * cp3y + t0**4 * cty;
            const cx1 = (1 - t1)**4 * csx + 4*(1 - t1)**3 * t1 * cp1x + 6*(1 - t1)**2 * t1**2 * cp2x + 4*(1 - t1) * t1**3 * cp3x + t1**4 * ctx;
            const cy1 = (1 - t1)**4 * csy + 4*(1 - t1)**3 * t1 * cp1y + 6*(1 - t1)**2 * t1**2 * cp2y + 4*(1 - t1) * t1**3 * cp3y + t1**4 * cty;
            arr[(i - 1) * stride + 0] = cx0;
            arr[(i - 1) * stride + 1] = cy0;
            arr[(i - 1) * stride + 6] = cx1;
            arr[(i - 1) * stride + 7] = cy1;
        }
    }
}

function createNDCMapperWithSize(bounds, margin = 0.05) {
    const { minX, maxX, minY, maxY } = bounds;
    const width = maxX - minX;
    const height = maxY - minY;

    const scale = Math.max(width, height) || 1;
    const offsetX = (scale - width) / 2;
    const offsetY = (scale - height) / 2;

    return function toNDCWithSize(x, y, size) {
        const normX = (x - minX + offsetX) / scale;
        const normY = (y - minY + offsetY) / scale;

        // 缩放到 [margin, 1 - margin]，再线性映射到 [-1, 1]
        const mappedX = (normX * (1 - 2 * margin) + margin) * 2 - 1;
        const mappedY = -((normY * (1 - 2 * margin) + margin) * 2 - 1);

        const s = size / scale * 2 * (1 - 2 * margin);
        return [mappedX, mappedY, s, s];
    };
}

function createNDCMapper(bounds, margin = 0.05) {
    const { minX, maxX, minY, maxY } = bounds;
    const width = maxX - minX;
    const height = maxY - minY;

    const scale = Math.max(width, height) || 1;
    const offsetX = (scale - width) / 2;
    const offsetY = (scale - height) / 2;

    return function toNDC(x, y) {
        const normX = (x - minX + offsetX) / scale;
        const normY = (y - minY + offsetY) / scale;

        const mappedX = (normX * (1 - 2 * margin) + margin) * 2 - 1;
        const mappedY = -((normY * (1 - 2 * margin) + margin) * 2 - 1);
        return [mappedX, mappedY];
    };
}

function getEquivalentNDCMatrix(bounds, margin = 0.05) {
    const { minX, maxX, minY, maxY } = bounds;
    const width = maxX - minX;
    const height = maxY - minY;

    const scale = Math.max(width, height) || 1;
    const offsetX = (scale - width) / 2;
    const offsetY = (scale - height) / 2;

    const a = (1 - 2 * margin) * 2 / scale;
    const bx = ((-minX + offsetX) / scale) * (1 - 2 * margin) + margin;
    const cx = bx * 2 - 1;

    const by = ((-minY + offsetY) / scale) * (1 - 2 * margin) + margin;
    const cy = -(by * 2 - 1);

    return [
        a, 0, 0,
        0, -a, 0,
        cx, cy, 1
    ]; // mat3 是列主序，可直接转 mat4 传 shader
}

function getGraphBounds(nodes) {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    nodes.forEach(node => {
        if (node.x < minX) minX = node.x;
        if (node.x > maxX) maxX = node.x;
        if (node.y < minY) minY = node.y;
        if (node.y > maxY) maxY = node.y;
    });

    return { minX, minY, maxX, maxY };
}

function getViewRectInWorldSpace(viewMatrix) {
    const inv = mat3.create();
    if (!mat3.invert(inv, viewMatrix)) return null;

    const cornersNDC = [
        [-1, -1], [1, -1], [1, 1], [-1, 1]
    ];
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (let [nx, ny] of cornersNDC) {
        const wx = inv[0] * nx + inv[3] * ny + inv[6];
        const wy = inv[1] * nx + inv[4] * ny + inv[7];
        minX = Math.min(minX, wx);
        minY = Math.min(minY, wy);
        maxX = Math.max(maxX, wx);
        maxY = Math.max(maxY, wy);
    }
    return { minX, minY, maxX, maxY };
}

function getViewRect(canvas, viewMatrix) {

    const topLeftWorld = canvasToWorld(0, 0, canvas.width, canvas.height, viewMatrix);
    const bottomRightWorld = canvasToWorld(canvas.width, canvas.height, canvas.width, canvas.height, viewMatrix);
    function NdcToPixel([x, y]) {
        return [
            x * canvas.width / 2 + canvas.width / 2,
            y * -canvas.height / 2 + canvas.height / 2
        ]
    }
    const [topLeftPixel_x, topLeftPixel_y] = NdcToPixel(topLeftWorld)
    const [bottomRightpixel_x, bottomRightpixel_y] = NdcToPixel(bottomRightWorld)
    const x = (topLeftPixel_x + bottomRightpixel_x) / 2
    const y = (topLeftPixel_y + bottomRightpixel_y) / 2
    const w = bottomRightpixel_x - topLeftPixel_x
    const h = topLeftPixel_y - bottomRightpixel_y
    const color = STYLE.viewRectColor// 半透明蓝色
    const pickColor = [0.0, 0.0, 0.0, 0.0];
    const selected = 0;
    return new Float32Array([
        x, y, w, h,
        ...color,
        ...pickColor,
        selected
    ]);
}

function getminiMapViewMatrix(bounds, paddingRatio = 0.1) {
    const { minX, maxX, minY, maxY } = bounds;
    const w = maxX - minX;
    const h = maxY - minY;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const scale = 2 / Math.max(w * (1 + paddingRatio * 2), h * (1 + paddingRatio * 2));

    const m = mat3.create();
    mat3.translate(m, m, [-cx, -cy]);       // ✅ 先平移到中心
    mat3.scale(m, m, [scale, scale]);       // ✅ 再整体缩放到 NDC 范围,顺序和主变换矩阵不同
    return m;
}


function applyViewMatrix(x, y, mat) {
    return [
        mat[0] * x + mat[3] * y + mat[6],
        mat[1] * x + mat[4] * y + mat[7]
    ];
}

function canvasToWorld(x_px, y_px, canvasWidth, canvasHeight, viewMatrix) {
    const x_ndc = (x_px / canvasWidth) * 2 - 1;
    const y_ndc = 1 - (y_px / canvasHeight) * 2;

    const ndc = vec2.fromValues(x_ndc, y_ndc);

    const invViewMatrix = mat3.create();
    mat3.invert(invViewMatrix, viewMatrix);

    const world = vec2.create();
    vec2.transformMat3(world, ndc, invViewMatrix);

    return world;
}

function matchColorID([r1, g1, b1], [r2, g2, b2]) {
    return decodeColorToIdFloat(r1, g1, b1) === decodeColorToIdFloat(r2, g2, b2);
}

function cloneData(data) {
    const cloned = {};
    for (const key in data) {
        if (Array.isArray(data[key])) {
            cloned[key] = data[key].slice(); // 浅拷贝数组
        } else if (data[key] instanceof Float32Array) {
            cloned[key] = new Float32Array(data[key]); // 浅拷贝 Float32Array
        } else {
            cloned[key] = data[key]; // 其他类型直接赋值
        }
    }
    return cloned;
}