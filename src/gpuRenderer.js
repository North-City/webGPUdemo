import { mat3, vec2 } from 'gl-matrix';
// === 全局样式配置 ===
const sampleCount = 4; // 开启 4 倍 MSAA
let msaaTexture = null; //  提前声明全局变量
// 字符渲染样式
const STYLE = {
    charSize: 0.015,             // 字体宽（世界坐标）
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

    // 箭头样式
    arrowColor: [0.3, 0.3, 0.3, 1.0],
    arrowSize: 0.03,             // 箭头大小
    charShiftY: -0.05,
    ringColor: [0.6, 0.6, 0.6, 0.6],
    ringInnerColor: [0.6, 0.2, 0.8, 0.0],
    ringHighlightColor: [0.6, 0.6, 0.6, 0.9],

    viewRectColor: [0.0, 0.0, 1.0, 0.1],

    nodeBound: null,
    minimapMargin: 0.05
};
let signal = {
    mouseDownID: 0,
}
let undoStack = []
let redoStack = []


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
    const { texture: fontTexture, sampler: fontSampler, uvMap } = await generateDigitTexture(device, chars);
    const imageList = ["../data/1.png"];
    const { canvas: imageCanvas, uvMap: imageUVMap } = await generateImageAtlas(imageList);
    const imageTexture = uploadTextureByImageData(device, imageCanvas);
    const imageSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });



    const data = extractDataFromG6(graph, canvas, uvMap, imageUVMap);

    //↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑图元需要的额外资源

    // === Uniforms
    const viewMatrix = mat3.create();
    const uniformBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bounds = getGraphBounds(graph.nodes);  // 获取坐标范围
    const miniMapViewMatrix = getEquivalentNDCMatrix(bounds, STYLE.minimapMargin);

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
            { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} }  // imageSamp
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
            { binding: 4, resource: imageSampler }
        ]
    });

    const miniMapBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: miniMapUniformBuffer } },
            { binding: 1, resource: fontTexture.createView() },
            { binding: 2, resource: fontSampler },
            { binding: 3, resource: imageTexture.createView() },
            { binding: 4, resource: imageSampler }
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
    out.color = vec4<f32>(1.0, 0.0, 0.0, 0.0); // 高亮
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
  let size = vec2<f32>(${STYLE.charSize}, ${STYLE.charSize * STYLE.charHeightRatio});
  let offset = input.pos * size;
  let world = input.center + offset;
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);

  out.uv = vec2<f32>(
    mix(input.u0, input.u1, input.pos.x * 0.5 + 0.5),
    mix(input.v0, input.v1, input.pos.y * -0.5 + 0.5)
  );
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

  // === 高亮判断 ===
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

  let isSelected = input.selected > 0.0;

  let shouldHighlight = isHover || isSelected;

  let baseColor = vec4<f32>(${STYLE.ringColor.join(", ")});
  let highlightColor = vec4<f32>(${STYLE.ringHighlightColor.join(", ")});

  out.color = select(baseColor, highlightColor, shouldHighlight);
  out.isHighlight = select(0.0, 1.0, shouldHighlight);
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
        let tex = textureSample(fontTex, fontSamp, input.uv);

        if (tex.a < 0.01) {
            discard; // ✅ 透明区域不绘制
        }

        var fontColor = vec4<f32>(${STYLE.charColor.join(", ")});

        if(input.selected > 0.0) {
            fontColor =  vec4<f32>(1.0, 0.0, 0.0, 1.0);
        } 
        
        return vec4<f32>(fontColor.rgb, fontColor.a * tex.a);
        }

        @vertex
fn image_vertex(input: ImageIn) -> Out {
  var out: Out;
  let world = input.instPos + input.pos * input.instSize;
  out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);
  out.uv = vec2<f32>(
    mix(input.u0, input.u1, input.pos.x+ 0.5),
    mix(input.v1, input.v0, input.pos.y+ 0.5)
  );
  out.color = vec4<f32>(1.0);
  return out;
}
@fragment
fn image_frag(input: Out) -> @location(0) vec4<f32> {
  return textureSample(imageTex, imageSamp, input.uv);
} 

@fragment
fn ring_frag(input: Out) -> @location(0) vec4<f32> {
  let r = length(input.uv);  // 当前像素到圆中心的归一化距离
  let outer = 0.4;
  let inner = outer - outer * input.strokeWidth;
  let highlightWidth = 0.1;
  let edgeWidth = 0.01;

  // 提前裁剪
  if (r > outer + highlightWidth + edgeWidth) {
    discard;
  }

  // ========== 圆环主区域 ========== //
  let tInner = smoothstep(inner - edgeWidth, inner + edgeWidth, r);
  let tOuter = 1.0 - smoothstep(outer - edgeWidth, outer + edgeWidth, r);
  let ringAlpha = tInner * tOuter;
  let baseColor = input.color;
  let baseAlpha = ringAlpha * baseColor.a;

  // ========== 外部模糊高亮带 ========== //
  let highlightColor = vec4<f32>(${STYLE.ringHighlightColor.join(", ")});
  let tHighlight = smoothstep(outer, outer + highlightWidth, r);
  let highlightAlpha = (1.0 - tHighlight) * input.isHighlight;

  // ========== 合成逻辑 ========== //
  var finalColor = baseColor;
  var finalAlpha = baseAlpha;

  // 只对 r ≥ inner 的区域执行颜色合成（防止中心发红）
  if (r >= inner) {
    finalColor = mix(baseColor, highlightColor, highlightAlpha);
    finalAlpha = max(baseAlpha, highlightAlpha * highlightColor.a);
  }

  return vec4(finalColor.rgb, finalAlpha);
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
            miniMapViewMatrix[0], miniMapViewMatrix[1], 0, 0,
            miniMapViewMatrix[3], miniMapViewMatrix[4], 0, 0,
            0, 0, 1, 0,
            miniMapViewMatrix[6], miniMapViewMatrix[7], 0, 1,
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

    canvas.addEventListener("click", async e => {

        if (signal.mouseDownID === 0) { console.log(13131); return }
        const select = true
        undoStack.push(signal.mouseDownID);
        redoStack = [];
        markSelectedById(signal.mouseDownID, data.rects, 13, -1, select);
        device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

        markSelectedById(signal.mouseDownID, data.ringInstances, 9, -1, select);
        device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

        markSelectedById(signal.mouseDownID, data.imageInstances, 13, -1, select);
        device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

        markSelectedById(signal.mouseDownID, data.charData, 11, -1, select);
        device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);

        signal.mouseDownID = 0; // 重置鼠标点击 ID
    });


    window.addEventListener("keydown", e => {
        if (e.ctrlKey && e.key === 'z') {
            if (!undoStack.length) return
            let id = undoStack.pop()
            redoStack.push(id)
            const select = false
            markSelectedById(id, data.rects, 13, -1, select);
            device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

            markSelectedById(id, data.ringInstances, 9, -1, select);
            device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

            markSelectedById(id, data.imageInstances, 13, -1, select);
            device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

            markSelectedById(id, data.charData, 11, -1, select);
            device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);
        } else if (e.ctrlKey && (e.key === 'y' || (e.shiftKey && e.key === 'Z'))) {
            if (!redoStack.length) return
            let id = redoStack.pop()
            undoStack.push(id)
            const select = true
            markSelectedById(id, data.rects, 13, -1, select);
            device.queue.writeBuffer(rectBuffer, 0, data.rects.buffer);

            markSelectedById(id, data.ringInstances, 9, -1, select);
            device.queue.writeBuffer(ringInstanceBuffer, 0, data.ringInstances.buffer);

            markSelectedById(id, data.imageInstances, 13, -1, select);
            device.queue.writeBuffer(imageInstanceBuffer, 0, data.imageInstances.buffer);

            markSelectedById(id, data.charData, 11, -1, select);
            device.queue.writeBuffer(charInstanceBuffer, 0, data.charData.buffer);
        }
    });
    canvas.addEventListener("mousedown", async e => {
        dragging = true; last = [e.clientX, e.clientY];
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
        signal.mouseDownID = id;
    });
    canvas.addEventListener("mouseup", () => {
        dragging = false
        //  signal.mouseDownID = 0;
    });
    canvas.addEventListener("mousemove", e => {
console.log("moveid",signal.mouseDownID);
        if (signal.mouseDownID !== 0) {
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

                arrowMoveById(signal.mouseDownID, data.arrowSegments, 12, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(arrowInstanceBuffer, 0, data.arrowSegments.buffer);

                edgeMoveById(signal.mouseDownID, data.polylines, 12, -1, [shiftX, shiftY]);
                device.queue.writeBuffer(lineBuffer, 0, data.polylines.buffer);


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
        if (dragging) {
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
    let dragging = false, last = [0, 0];
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

        pass.setPipeline(rectPipeline);
        pass.setVertexBuffer(0, quadBuffer);
        pass.setVertexBuffer(1, rectBuffer);
        pass.draw(4, data.rects.length / 13);

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
    const padding = 20;
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

            const u0 = x / canvas.width;
            const u1 = (x + item.width) / canvas.width;
            const v0 = rowIndex * cellHeights / canvas.height;
            const v1 = (rowIndex * cellHeights + cellHeights) / canvas.height;

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
    // const canvas = generateDigitTextureCanvas(); // 你已有的图集绘制逻辑
    const { canvas, uvMap } = generateCharTextureCanvas(charList)
    const texture = uploadTextureByImageData(device, canvas);
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    return { texture, sampler, uvMap };
}

function uploadTextureByImageData(device, canvas) {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const rgba = new Uint8Array(imageData.data);

    const bytesPerPixel = 4;
    const unpaddedBytesPerRow = canvas.width * bytesPerPixel;
    const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;

    const paddedData = new Uint8Array(paddedBytesPerRow * canvas.height);

    // 每行拷贝原始数据 → paddedData
    for (let y = 0; y < canvas.height; y++) {
        const srcOffset = y * unpaddedBytesPerRow;
        const dstOffset = y * paddedBytesPerRow;
        paddedData.set(rgba.subarray(srcOffset, srcOffset + unpaddedBytesPerRow), dstOffset);
    }

    const texture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    device.queue.writeTexture(
        { texture },
        paddedData,
        {
            bytesPerRow: paddedBytesPerRow,
            rowsPerImage: canvas.height
        },
        [canvas.width, canvas.height]
    );
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
    let nodeColorIdMap = new Map();
    graph.nodes.forEach((node, index) => {
        const nodeID = encodeIdToColor(index + 1); console.log(nodeID);
        nodeColorIdMap.set(node.id, nodeID);
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
        const charSpacing = charSize * STYLE.charSpacingRatio;
        const step = charSize + charSpacing;
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

    graph.edges.forEach(edge => {
        edge.sourceColorId = nodeColorIdMap.get(edge.source) || encodeIdToColor(0);
        edge.targetColorId = nodeColorIdMap.get(edge.target) || encodeIdToColor(0);
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
            polylines.push(mid1_X, mid1_Y, ...edge.sourceColorId, mid2_X, mid2_Y, ...edge.targetColorId);
            polylines_mini.push(edge.startPoint.x - loopoffsetX, edge.startPoint.y - loopoffsetY, ...edge.sourceColorId, edge.endPoint.x + loopoffsetX, edge.endPoint.y - loopoffsetY, ...edge.targetColorId);
            polylines.push(mid2_X, mid2_Y, ...edge.sourceColorId, targetX, targetY, ...edge.targetColorId);
            polylines_mini.push(edge.endPoint.x + loopoffsetX, edge.endPoint.y - loopoffsetY, ...edge.sourceColorId, edge.endPoint.x, edge.endPoint.y, ...edge.targetColorId);
            arrows.push(mid2_X, mid2_Y, targetX, targetY, ...edge.sourceColorId, ...edge.targetColorId);
        } else {
            const [x1, y1] = scale(edge.startPoint.x, edge.startPoint.y);
            const [x2, y2] = scale(edge.endPoint.x, edge.endPoint.y);
            polylines.push(x1, y1, ...edge.sourceColorId, x2, y2, ...edge.targetColorId);
            arrows.push(x1, y1, x2, y2, ...edge.sourceColorId, ...edge.targetColorId);
            polylines_mini.push(edge.startPoint.x, edge.startPoint.y, ...edge.sourceColorId, edge.endPoint.x, edge.endPoint.y, ...edge.targetColorId);
            // const [x1_mini, y1_mini] = toNDC(edge.startPoint.x, edge.startPoint.y);
            // const [x2_mini, y2_mini] = toNDC(edge.endPoint.x, edge.endPoint.y);
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
function markSelectedById(id, arr, stride, selectedOffset, select = true) {
    for (let i = 1; i * stride <= arr.length; i++) {
        let [r, g, b] = [arr[i * stride + selectedOffset - 4], arr[i * stride + selectedOffset - 3], arr[i * stride + selectedOffset - 2]]
        if (decodeColorToIdFloat(r, g, b) === id) {
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
        if (decodeColorToIdFloat(r, g, b) === id) {
            arr[(i - 1) * stride] += shiftX
            arr[(i - 1) * stride + 1] += shiftY
        }
    }
}

function arrowMoveById(id, arr, stride, selectedOffset, [shiftX, shiftY]) {
    for (let i = 1; i * stride <= arr.length; i++) {
        let [rSource, gSource, bSource] = [arr[i * stride + selectedOffset - 7], arr[i * stride + selectedOffset - 6], arr[i * stride + selectedOffset - 5]]
        let [rTarget, gTarget, bTarget] = [arr[i * stride + selectedOffset - 3], arr[i * stride + selectedOffset - 2], arr[i * stride + selectedOffset - 1]]
        const sourceId = decodeColorToIdFloat(rSource, gSource, bSource);
        const targetId = decodeColorToIdFloat(rTarget, gTarget, bTarget);
        console.log("sourceId", [rSource, gSource, bSource], "targetId", [rTarget, gTarget, bTarget], "id", id);

        if (sourceId === id) {
            arr[(i - 1) * stride] += shiftX
            arr[(i - 1) * stride + 1] += shiftY
        }
        if (targetId === id) {
            arr[(i - 1) * stride + 2] += shiftX
            arr[(i - 1) * stride + 3] += shiftY
        }
    }
}

function edgeMoveById(id, arr, stride, selectedOffset, [shiftX, shiftY]) {
    for (let i = 1; i * stride <= arr.length; i++) {
        let [rSource, gSource, bSource] = [arr[i * stride + selectedOffset - 9], arr[i * stride + selectedOffset - 8], arr[i * stride + selectedOffset - 7]]
        let [rTarget, gTarget, bTarget] = [arr[i * stride + selectedOffset - 3], arr[i * stride + selectedOffset - 2], arr[i * stride + selectedOffset - 1]]
        const sourceId = decodeColorToIdFloat(rSource, gSource, bSource);
        const targetId = decodeColorToIdFloat(rTarget, gTarget, bTarget);
        console.log("sourceId",sourceId, [rSource, gSource, bSource], "targetId",targetId, [rTarget, gTarget, bTarget], "id", id);

        if (sourceId === id) {
            arr[(i - 1) * stride] += shiftX
            arr[(i - 1) * stride + 1] += shiftY
        }
        if (targetId === id) {
            arr[(i - 1) * stride + 6] += shiftX
            arr[(i - 1) * stride + 7] += shiftY
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

function getMiniMapViewMatrix(bounds, paddingRatio = 0.1) {
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