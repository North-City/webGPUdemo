import { mat3 } from 'gl-matrix';
// === 全局样式配置 ===
const sampleCount = 4; // 开启 4 倍 MSAA
let msaaTexture = null; //  提前声明全局变量
// 字符渲染样式
const STYLE = {
    charSize: 0.015,             // 字体宽（世界坐标）
    charHeightRatio: 1.5,        // 字体高宽比
    charSpacingRatio: 0.1,       // 字符之间的空隙（占charSize的比例）
    clearColor: [0.9, 0.9, 0.9, 1],//background
    charColor: [0.0, 0.0, 0.0, 0.9],  // 字体颜色

    // 节点矩形样式
    nodeColor: [0.6, 0.8, 0.8, 0.6],

    // 连线样式
    edgeColor: [0.5, 0.5, 0.5, 0.5],

    // 箭头样式
    arrowColor: [0.3, 0.3, 0.3, 1.0],
    arrowSize: 0.03,             // 箭头大小
    charShiftY: 0.02
};
export async function initWebGPU(graph) {
    if (!navigator.gpu) {
        alert("WebGPU not supported");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const limits = adapter.limits;
    console.log("maxSampleCount:", limits.maxSampledTexturesPerShaderStage);
    const device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", e => {
        console.error("GPU ERROR:", e.error);
    });
    const canvas = document.getElementById("webgpuCanvas");
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({ device, format, alphaMode: "opaque" });

    const chars = collectCharsFromGraph(graph)
    const UVmap = createCharUVMap(chars)
    const data = extractDataFromG6(graph, canvas, UVmap);
    const { texture: fontTexture, sampler: fontSampler } = await generateDigitTexture(device, chars);

    // === Uniforms
    const viewMatrix = mat3.create();
    const uniformBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
        ]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: fontTexture.createView() },
            { binding: 2, resource: fontSampler }
        ]
    });

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

      struct Out {
        @builtin(position) position : vec4<f32>,
        @location(0) color : vec4<f32>,
        @location(1) uv : vec2<f32>
      };

      struct VertexIn {
        @location(0) pos: vec2<f32>,
        @location(1) instPos: vec2<f32>,
        @location(2) instSize: vec2<f32>,
        @location(3) instColor: vec4<f32>,
      };

      struct SimpleIn {
        @location(0) pos: vec2<f32>,
      };

      struct ArrowIn {
        @location(0) pos: vec2<f32>,
        @location(1) p1: vec2<f32>,
        @location(2) p2: vec2<f32>,
      };

      struct CharIn {
        @location(0) pos: vec2<f32>,
        @location(1) center: vec2<f32>,
        @location(2) u0: f32,
        @location(3) u1: f32
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
    out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // 高亮
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
        let size = vec2<f32>(${STYLE.charSize}, ${STYLE.charSize * STYLE.charHeightRatio}); // 字符尺寸size 就是单个字符在世界坐标下的尺寸。
        let offset = input.pos * size;
        let world = input.center + offset;
        out.position = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);

        // 这里修正 UV 映射
        out.uv = vec2<f32>(
            mix(input.u0, input.u1, input.pos.x * 0.5 + 0.5), // 横向 UV 正确在 u0 ~ u1 区间变化
            input.pos.y * -0.5 + 0.5 // 纵向 UV: 上 0，下 1（因为canvas坐标是左上角）
        );
        return out;
        }

        @fragment
        fn fragment_main(input: Out) -> @location(0) vec4<f32> {
        return input.color; // 用于矩形、边、箭头
        }

        @fragment
        fn char_frag(input: Out) -> @location(0) vec4<f32> {
        let tex = textureSample(fontTex, fontSamp, input.uv);

        if (tex.a < 0.01) {
            discard; // ✅ 透明区域不绘制
        }

        let fontColor = vec4<f32>(${STYLE.charColor.join(", ")}); // 白色字
        return vec4<f32>(fontColor.rgb, fontColor.a * tex.a);
        }
    `
    });

    // === 创建各类 buffer（矩形、线段、箭头、字符）

    const quad = new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]);
    const quadBuffer = createBuffer(device, quad, GPUBufferUsage.VERTEX);

    const rectBuffer = createBuffer(device, data.rects, GPUBufferUsage.VERTEX);
    const lineBuffer = createBuffer(device, data.polylines, GPUBufferUsage.VERTEX);
    const arrowVertexBuffer = createBuffer(device, new Float32Array([
        0.0, 0.0,
        -STYLE.arrowSize, STYLE.arrowSize * 0.4,
        -STYLE.arrowSize, -STYLE.arrowSize * 0.4
    ]), GPUBufferUsage.VERTEX);
    const arrowInstanceBuffer = createBuffer(device, data.arrowSegments, GPUBufferUsage.VERTEX);
    const charQuadBuffer = createBuffer(device, new Float32Array([
        -0.5, -0.5, 0.5, -0.5,
        -0.5, 0.5, 0.5, 0.5
    ]), GPUBufferUsage.VERTEX);
    const charInstanceBuffer = createBuffer(device, data.charData, GPUBufferUsage.VERTEX);

    // === 创建 pipelines（矩形、边、箭头、字符）
    const rectPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "rect_vertex", "fragment_main", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 32, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32x2", offset: 8 },
                { shaderLocation: 3, format: "float32x4", offset: 16 }
            ]
        }
    ], "triangle-strip");

    const linePipeline = createPipeline(device, shaderModule, pipelineLayout, format, "simple_vertex", "fragment_main", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] }
    ], "line-list");

    const arrowPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "arrow_vertex", "fragment_main", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 16, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32x2", offset: 8 }
            ]
        }
    ], "triangle-list");

    const charPipeline = createPipeline(device, shaderModule, pipelineLayout, format, "char_vertex", "char_frag", [
        { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, format: "float32x2", offset: 0 }] },
        {
            arrayStride: 16, stepMode: "instance", attributes: [
                { shaderLocation: 1, format: "float32x2", offset: 0 },
                { shaderLocation: 2, format: "float32", offset: 8 },
                { shaderLocation: 3, format: "float32", offset: 12 }
            ]
        }
    ], "triangle-strip");

    // === 控制视图变换
    let scale = 1;
    let offset = [0, 0];
    let hoverPos1 = [999, 999]; // 默认值设置为图外
    let hoverPos2 = [999, 999]; // 默认值设置为图外
    const hoverRadius = 0.03;
    const updateMatrix = () => {
        mat3.identity(viewMatrix);
        mat3.translate(viewMatrix, viewMatrix, offset);
        mat3.scale(viewMatrix, viewMatrix, [scale, scale]);
        const mat4 = new Float32Array([
            viewMatrix[0], viewMatrix[1], 0, 0,
            viewMatrix[3], viewMatrix[4], 0, 0,
            0, 0, 1, 0,
            viewMatrix[6], viewMatrix[7], 0, 1,
            hoverPos1[0], hoverPos1[1], hoverPos2[0], hoverPos2[1] // 更新为两个坐标
        ]);
        console.log(mat4);
        
        device.queue.writeBuffer(uniformBuffer, 0, mat4);
    };

    // === 交互
    const hoverSize = 0.01;
    canvas.addEventListener("wheel", e => {
        e.preventDefault();
        
        // 1. 获取鼠标的NDC坐标（WebGPU坐标系：Y向上）
        const rect = canvas.getBoundingClientRect();
        const ndcX = (e.clientX - rect.left) / canvas.width * 2 - 1;
        const ndcY = 1 - (e.clientY - rect.top) / canvas.height * 2;
        // console.log(ndcX,ndcY);
        
    
        // 2. 计算当前鼠标的世界坐标
        const invView = mat3.create();
        if (!mat3.invert(invView, viewMatrix)) return;
        const worldX = invView[0] * ndcX + invView[3] * ndcY + invView[6];
        const worldY = invView[1] * ndcX + invView[4] * ndcY + invView[7];
        // console.log(worldX,worldY);
        
        // 3. 使用加法缩放（更稳定）
        const zoomStep = 0.1 * Math.log(scale + 1); // 动态步长（随scale增大而减小）
        let newScale = e.deltaY < 0 ? scale + zoomStep : scale - zoomStep;
        newScale = Math.min(Math.max(newScale, 0.05), 20);
    
        // 4. 精确锚点补偿
        offset[0] -= (worldX - offset[0]) * (newScale / scale - 1);
        offset[1] -= (worldY - offset[1]) * (newScale / scale - 1);
        scale = newScale;
    
        updateMatrix();
    });
    
    
    
    canvas.addEventListener("mousedown", e => { dragging = true; last = [e.clientX, e.clientY]; });
    canvas.addEventListener("mouseup", () => dragging = false);
    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width * 2 - 1;
        const y = 1 - (e.clientY - rect.top) / canvas.height * 2;
        const inv = mat3.create();
        if (mat3.invert(inv, viewMatrix)) {
            const worldX = inv[0] * x + inv[3] * y + inv[6];
            const worldY = inv[1] * x + inv[4] * y + inv[7];

            hoverPos1 = [worldX - hoverSize / 2, worldY + hoverSize / 2]; // 左上角
            hoverPos2 = [worldX + hoverSize / 2, worldY - hoverSize / 2]; // 右下角
        } else {
            hoverPos1 = [999, 999];
            hoverPos2 = [999, 999];
        }
        // 如果在拖动，也更新偏移
        if (dragging) {
            const dx = (e.clientX - last[0]) / 400;
            const dy = (e.clientY - last[1]) / 300;
            offset[0] += dx;
            offset[1] -= dy;
            last = [e.clientX, e.clientY];
        }
        updateMatrix();
    });
    let dragging = false, last = [0, 0];
    updateMatrix();

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
        pass.draw(data.polylines.length / 2);
    
        pass.setPipeline(rectPipeline);
        pass.setVertexBuffer(0, quadBuffer);
        pass.setVertexBuffer(1, rectBuffer);
        pass.draw(4, data.rects.length / 8);
    
        pass.setPipeline(arrowPipeline);
        pass.setVertexBuffer(0, arrowVertexBuffer);
        pass.setVertexBuffer(1, arrowInstanceBuffer);
        pass.draw(3, data.arrowSegments.length / 4);
    
        pass.setPipeline(charPipeline);
        pass.setVertexBuffer(0, charQuadBuffer);
        pass.setVertexBuffer(1, charInstanceBuffer);
        pass.draw(4, data.charData.length / 4);
    
        pass.end();
        device.queue.submit([encoder.finish()]);
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

function createPipeline(device, module, layout, format, vert, frag, buffers, topology) {
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
    ctx.font = "24px monospace";//字在图集中的大小越大，图集的字就越清晰，占的纹理空间也越多（可能会让单个字符图占据更大的纹理区域，UV计算不变）
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
    const canvas = document.createElement("canvas");
    const fontSize = 72;
    const cellW = fontSize + 8; // 给字体大小预留一点padding
    const cellH = fontSize * 1.5;
    canvas.width = cellW * charList.length;
    canvas.height = cellH;

    const ctx = canvas.getContext("2d", { alpha: true });
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.font = `${fontSize}px monospace`;//字在图集中的大小越大，图集的字就越清晰，占的纹理空间也越多（可能会让单个字符图占据更大的纹理区域，UV计算不变）
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < charList.length; i++) {
        const x = i * cellW + cellW / 2;
        const y = cellH / 2;
        ctx.fillText(charList[i], x, y);
    }

    return canvas;
}


async function generateDigitTexture(device, charList) {
    // const canvas = generateDigitTextureCanvas(); // 你已有的图集绘制逻辑
    const canvas = generateCharTextureCanvas(charList)
    const texture = uploadTextureByImageData(device, canvas);
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    return { texture, sampler };
}

function uploadTextureByImageData(device, canvas) {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const rgba = new Uint8Array(imageData.data);

    const texture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    device.queue.writeTexture(
        { texture },
        rgba,
        { bytesPerRow: canvas.width * 4 },
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


function extractDataFromG6(graph, canvas, uvMap) {
    function scale(px, py) {
        return [(px / canvas.width) * 2 - 1, 1 - (py / canvas.height) * 2];
    }

    let rects = [], polylines = [], arrows = [], chars = [];

    graph.nodes.forEach(node => {
        const [x, y] = scale(node.x, node.y);
        const w = node.size[0] / canvas.width * 2;
        const h = node.size[1] / canvas.height * 2;
        // const w = node.width / canvas.width * 2;//for GPU
        // const h = node.height / canvas.height * 2;
        rects.push(x, y, w, h, ...STYLE.nodeColor);
        if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(w) || Number.isNaN(h) || w == 0 || h == 0) {
            console.log(x, y, w, h);

        }


        const idStr = node.label.toString();
        const charSize = STYLE.charSize;
        const charSpacing = charSize * STYLE.charSpacingRatio;
        const step = charSize + charSpacing;
        const baseX = x - (idStr.length - 1) * step * 0.5;//字符相对节点中心位置的偏移

        for (let i = 0; i < idStr.length; i++) {
            const ch = idStr[i];
            const [u0, u1] = uvMap[ch] || [0, 0.1];
            chars.push(baseX + i * step, y + STYLE.charShiftY, u0, u1);//含有字符相对节点位置的y偏移
        }

    });

    graph.edges.forEach(edge => {
        const [x1, y1] = scale(edge.startPoint.x, edge.startPoint.y);
        const [x2, y2] = scale(edge.endPoint.x, edge.endPoint.y);
        polylines.push(x1, y1, x2, y2);
        arrows.push(x1, y1, x2, y2);
    });

    return {
        rects: new Float32Array(rects),
        polylines: new Float32Array(polylines),
        arrowSegments: new Float32Array(arrows),
        charData: new Float32Array(chars)
    };
}

