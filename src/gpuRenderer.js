import { mat3 } from 'gl-matrix';



export async function initWebGPU(G6Graph) {
    if (!navigator.gpu) {
        alert("WebGPU not supported");
        return;
    }
    // G6Graph = await testData()

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpuCanvas");
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    let data = extractDataFromG6(G6Graph, canvas)

    context.configure({
        device,
        format,
        alphaMode: 'opaque'
    });

    // === Uniform buffer + shared layout
    const viewMatrix = mat3.create();
    const uniformBuffer = device.createBuffer({
        size: 64, // mat4x4<f32>
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: "uniform" }
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    // === Rectangle geometry (instanced)
    const quad = new Float32Array([
        -0.5, -0.5,
        0.5, -0.5,
        -0.5, 0.5,
        0.5, 0.5
    ]);
    const quadBuffer = device.createBuffer({
        size: quad.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    new Float32Array(quadBuffer.getMappedRange()).set(quad);
    quadBuffer.unmap();

    // const rects = new Float32Array([
    //     -0.4, 0.3, 0.2, 0.2, 1, 0, 0, 1,
    //     0.4, 0.3, 0.2, 0.2, 0, 1, 0, 1,
    //     -0.4, -0.3, 0.2, 0.2, 0, 0, 1, 1,
    //     0.4, -0.3, 0.2, 0.2, 1, 1, 0, 1
    // ]);
    const rects = new Float32Array(data.rects)
    const rectBuffer = device.createBuffer({
        size: rects.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(rectBuffer, 0, rects);

    // === Polyline & arrow geometry
    // const polyline = new Float32Array([
    //     0, 0.2,
    //     0.5, 1.0,
    //     0.5, 0.5,
    //     -1, -1
    // ]);
    const polyline = new Float32Array(data.polylines)
    const lineBuffer = device.createBuffer({
        size: polyline.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(lineBuffer, 0, polyline);

    // const arrowSegments = new Float32Array([
    //     0.0, 0.0, -0.5, 0.5,   // 线段1（起点→终点）
    //     -0.3, -0.3, 0.0, 0.0    // 线段2（起点→终点）
    // ]);
    const arrowSegments = new Float32Array(data.arrowSegments)
    const arrowVertices = new Float32Array([
        0.0, 0.0,     // 尖端
        -0.05, 0.02,   // 上边
        -0.05, -0.02    // 下边
    ]);

    const arrowVertexBuffer = device.createBuffer({
        size: arrowVertices.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    new Float32Array(arrowVertexBuffer.getMappedRange()).set(arrowVertices);
    arrowVertexBuffer.unmap();

    const arrowInstanceBuffer = device.createBuffer({
        size: arrowSegments.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(arrowInstanceBuffer, 0, arrowSegments);


    // === Shared shader
    const shaderModule = device.createShaderModule({
        code: `
      struct Uniforms {
        viewMatrix : mat4x4<f32>
      };
      @group(0) @binding(0) var<uniform> uniforms : Uniforms;

      struct Out {
        @builtin(position) position : vec4<f32>,
        @location(0) color : vec4<f32>,
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

      @vertex
      fn rect_vertex(input: VertexIn) -> Out {
        var out: Out;
        let world = input.instPos + input.pos * input.instSize;
        let transformed = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);
        out.position = transformed;
        out.color = input.instColor;
        return out;
      }

      @vertex
      fn simple_vertex(input: SimpleIn) -> Out {
        var out: Out;
        let transformed = uniforms.viewMatrix * vec4<f32>(input.pos, 0.0, 1.0);
        out.position = transformed;
        out.color = vec4<f32>(0.7, 0.7, 0.7, 1.0);
        return out;
      }

      @fragment
      fn fragment_main(input: Out) -> @location(0) vec4<f32> {
        return input.color;
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
  let transformed = uniforms.viewMatrix * vec4<f32>(world, 0.0, 1.0);

  out.position = transformed;
  out.color = vec4<f32>(0.7, 0.7, 0.7, 1.0);
  return out;
}
    `
    });

    // === Pipelines (with explicit layout)
    const rectPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: "rect_vertex",
            buffers: [
                { arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] },
                {
                    arrayStride: 32, stepMode: "instance", attributes: [
                        { shaderLocation: 1, offset: 0, format: "float32x2" },
                        { shaderLocation: 2, offset: 8, format: "float32x2" },
                        { shaderLocation: 3, offset: 16, format: "float32x4" }
                    ]
                }
            ]
        },
        fragment: { module: shaderModule, entryPoint: "fragment_main", targets: [{ format }] },
        primitive: { topology: "triangle-strip" }
    });

    const linePipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: "simple_vertex",
            buffers: [{ arrayStride: 8, stepMode: "vertex", attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }]
        },
        fragment: { module: shaderModule, entryPoint: "fragment_main", targets: [{ format }] },
        primitive: { topology: "line-list" }
    });

    const arrowPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: "arrow_vertex", // ✅ 使用新的顶点函数
            buffers: [
                {
                    arrayStride: 8,
                    stepMode: "vertex",
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x2" } // 三角形形状顶点
                    ]
                },
                {
                    arrayStride: 16,
                    stepMode: "instance",
                    attributes: [
                        { shaderLocation: 1, offset: 0, format: "float32x2" }, // 起点 p1
                        { shaderLocation: 2, offset: 8, format: "float32x2" }  // 终点 p2
                    ]
                }
            ]
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fragment_main",
            targets: [{ format }]
        },
        primitive: { topology: "triangle-list" }
    });

    // === View Matrix update
    let scale = 1;
    let offset = [0, 0];
    function updateMatrix() {
        mat3.identity(viewMatrix);
        mat3.translate(viewMatrix, viewMatrix, offset);
        mat3.scale(viewMatrix, viewMatrix, [scale, scale]);

        const mat4 = new Float32Array([
            viewMatrix[0], viewMatrix[1], 0, 0,
            viewMatrix[3], viewMatrix[4], 0, 0,
            0, 0, 1, 0,
            viewMatrix[6], viewMatrix[7], 0, 1
        ]);
        device.queue.writeBuffer(uniformBuffer, 0, mat4);
    }

    // === Interaction
    let dragging = false, last = [0, 0];
    canvas.addEventListener("mousedown", e => {
        dragging = true;
        last = [e.clientX, e.clientY];
    });
    canvas.addEventListener("mouseup", () => dragging = false);
    canvas.addEventListener("mousemove", e => {
        if (!dragging) return;
        const dx = (e.clientX - last[0]) / 400;
        const dy = (e.clientY - last[1]) / 300;
        offset[0] += dx;
        offset[1] -= dy;
        last = [e.clientX, e.clientY];
        updateMatrix();
    });
    canvas.addEventListener("wheel", e => {
        e.preventDefault();
        scale *= 1 - e.deltaY * 0.001;
        updateMatrix();
    });

    updateMatrix();

    function frame() {
        const encoder = device.createCommandEncoder();
        const view = context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view,
                loadOp: "clear",
                storeOp: "store",
                clearValue: [1.0, 1.0, 1.0, 1.0]
            }]
        });

        pass.setBindGroup(0, bindGroup);
        pass.setPipeline(linePipeline);
        pass.setVertexBuffer(0, lineBuffer);
        pass.draw(polyline.length / 2);

        pass.setPipeline(rectPipeline);
        pass.setVertexBuffer(0, quadBuffer);
        pass.setVertexBuffer(1, rectBuffer);
        pass.draw(4, rects.length / 8);


        pass.setPipeline(arrowPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.setVertexBuffer(0, arrowVertexBuffer);
        pass.setVertexBuffer(1, arrowInstanceBuffer);
        pass.draw(3, arrowSegments.length / 4); // 每两个点一条线段（1个箭头）

        pass.end();
        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }

    frame();
}
function extractDataFromG6(graph, canvas) {
    // const rects = new Float32Array([
    //     -0.4, 0.3, 0.2, 0.2, 1, 0, 0, 1,
    //     0.4, 0.3, 0.2, 0.2, 0, 1, 0, 1,
    //     -0.4, -0.3, 0.2, 0.2, 0, 0, 1, 1,
    //     0.4, -0.3, 0.2, 0.2, 1, 1, 0, 1
    // ]);
    // const polyline = new Float32Array([
    //     0, 0.2,
    //     0.5, 1.0,
    //     0.5, 0.5,
    //     -1, -1
    // ]);
    // const arrowSegments = new Float32Array([
    //     0.0, 0.0, -0.5, 0.5,   // 线段1（起点→终点）
    //     -0.3, -0.3, 0.0, 0.0    // 线段2（起点→终点）
    // ]);
    let rects = []
    let edges = []
    function scaleRectToGpu(px, py, pw, ph, canvas) {
        let ndcX = (px / canvas.width) * 2 - 1
        let ndcY = 1 - (py / canvas.height) * 2
        let ndcWidth = (pw / canvas.width) * 2
        let ndcHeight = (ph / canvas.height) * 2
        return [ndcX, ndcY, ndcWidth, ndcHeight]
    }
    function scalePointToGpu(px1, py1, px2, py2,canvas) {
        let ndcX1 = (px1 / canvas.width) * 2 - 1
        let ndcY1 = 1 - (py1 / canvas.height) * 2
        let ndcX2 = (px2 / canvas.width) * 2 - 1
        let ndcY2 = 1 - (py2 / canvas.height) * 2
        return [ndcX1, ndcY1, ndcX2, ndcY2]
    }
    graph.cfg.data.nodes.forEach(element => {
        const [ndcX, ndcY, ndcWidth, ndcHeight] = scaleRectToGpu(element.x, element.y, element.size[0], element.size[1], canvas)
        rects =rects.concat([ndcX, ndcY, ndcWidth, ndcHeight,0.6, 0.8,0.8, 1])
    });
    graph.cfg.data.edges.forEach(element => {
        let x1 = element.startPoint.x
        let y1 = element.startPoint.y
        let x2 = element.endPoint.x
        let y2 = element.endPoint.y
        const [ndcX1, ndcY1,ndcX2, ndcY2] = scalePointToGpu(x1,y1, x2,y2,canvas)
        edges =edges.concat([ndcX1, ndcY1,ndcX2, ndcY2])
    });
    let result = {
        rects: rects,
        polylines: edges,
        arrowSegments: edges
    }
    console.log(result);

    return result
}

function testData() {
    let testTime = 10000
    let rects = []
    for (let i = 0; i < testTime; i++) {
        rects = rects.concat([0.0, 0.0 - 0.3 * i, 0.2, 0.2, 1, 0, 0, 1])
    }
    let polylines = []
    for (let i = 0; i < testTime; i++) {
        polylines = polylines.concat([0.1, 0.0 - 0.3 * i, 0.5, 0.0 - 0.3 * i])
    }
    let arrowSegments = polylines
    // for (let i = 0; i < testTime; i++) {
    //     arrowSegments = arrowSegments.concat([0.1, 0.0 - 0.3 * i, 0.5, 0.0 - 0.3 * i])
    // }
    console.log(rects);

    let result = {
        rects: rects,
        polylines: polylines,
        arrowSegments: arrowSegments
    }
    // console.log(result);

    return result
}