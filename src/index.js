import { initWebGPU } from "./gpuRenderer";
import { dagreLayout } from "./dagreGenerater";



import './style.css';
import * as G6 from '@antv/g6';
import { Renderer as SVGRenderer } from '@antv/g-svg';
import { Renderer as WebGLRenderer } from '@antv/g-webgl';
import originData from '../data/data-11W.csv';
// import dummyData from "../data/data_11W_dummy.json"
import data1 from '../data/data_one'
import data2 from '../data/data_two'

//data init
let attrList = originData.shift()
console.log(attrList);
function replaceAsterisks(arr) {
  return arr.map(subArr => {
    // 复制数组，避免修改原始数据
    let newArr = [...subArr];

    // 替换第三列（索引 2）和第六列（索引 5）中所有连续的 `*` 为 `~`
    if (typeof newArr[2] === "string") {
      newArr[2] = newArr[2].replace(/\*+/g, "~");
    }
    if (typeof newArr[5] === "string") {
      newArr[5] = newArr[5].replace(/\*+/g, "~");
    }

    return newArr;
  });
}
let dat = originData
function mergeAndDeduplicate(arr) {
  // 提取每个子数组的第三项（索引2）和第六项（索引5）
  let merged = arr.flatMap(subArr => [subArr[2], subArr[5]]);

  // 过滤掉 undefined 值（如果某些子数组的长度不够）
  merged = merged.filter(item => item !== undefined);

  // 使用 Set 去重，并转换回数组
  return [...new Set(merged)];
}
function generateEdges(data) {
  let edgeMap = new Map();

  data.forEach(subArr => {
    let source = subArr[2];  // 假设第一项是 source node
    let target = subArr[5];  // 第六项是 target node
    let amount = parseFloat(subArr[6]);  // 第七项是转出金额

    if (source && target && amount) {
      let key = `${source}&${target}`;
      edgeMap.set(key, (edgeMap.get(key) || 0) + amount);
    }
  });

  // 转换 Map 为数组
  let edges = Array.from(edgeMap, ([key, label]) => {
    let [source, target] = key.split("&");
    return {
      source, target
      // , label 
    };
  });

  return edges;
}
let cardID = mergeAndDeduplicate(dat)
let eList = generateEdges(dat)


let data
const show = "large"
switch (show) {
  case "large":
    data = {
      nodes: cardID.map(d => {
        return {
          id: d,
          label: d
        }
      }),
      edges: eList,
    }
    break;
  case "JA1":
    data = data1
    break;
  case "JA2":
    data = data2
    break;
  case "dummy":
    data = dummyData
    break
  case "makeDummy":
    data = {
      nodes: cardID.map(d => ({
        id: d,
        label: d
      })),
      edges: eList,
    };

    // === 封装函数：添加假节点和假边 ===
    function augmentGraph(data, dummyNodeCount, dummyEdgeCount) {
      const existingNodeIds = data.nodes.map(n => n.id);

      // 生成假节点
      const dummyNodes = Array.from({ length: dummyNodeCount }, (_, i) => {
        const id = `dummy_${i}`;
        return { id, label: id };
      });

      // 合并所有节点 id（用于生成边）
      const allNodeIds = existingNodeIds.concat(dummyNodes.map(n => n.id));

      // 生成假边
      const dummyEdges = [];
      for (let i = 0; i < dummyEdgeCount; i++) {
        let source, target;
        do {
          source = allNodeIds[Math.floor(Math.random() * allNodeIds.length)];
          target = allNodeIds[Math.floor(Math.random() * allNodeIds.length)];
        } while (source === target); // 避免自环边

        dummyEdges.push({ source, target });
      }

      // 合并进原始数据
      data.nodes.push(...dummyNodes);
      data.edges.push(...dummyEdges);
    }

    // === 调用函数添加点边 ===
    augmentGraph(data, 25, 2300);
    function downloadJSON(data, filename = "graph_data.json") {
  const jsonStr = JSON.stringify(data, null, 2); // 美化格式缩进
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url); // 清理 URL 对象
}

// 调用导出
downloadJSON(data);
    break

}



const graph = new G6.Graph({
  container: 'container',
  width: 1000,
  height: 1000,
  // fitView: true,
  modes: {
    default: ['drag-canvas', 'zoom-canvas']//, 'drag-node'
  },
  renderer: () => new SVGRenderer(),
  layout: {
    type: 'dagre',
    rankdir: 'TB',
    align: 'DL',
    nodesepFunc: () => {
      return 1;
    },
    ranksepFunc: () => {
      return 1;
    }
  },
  animate: false,
  defaultNode: {
    size: [50, 20],
    type: 'rect',
    style: {
      lineWidth: 2,
      stroke: '#5B8FF9',
      fill: '#C6E5FF',
      fillOpacity: 0.1
    }
  },
  defaultEdge: {
    type: 'cubic-horizontal',
    size: 1,
    color: '#e2e2e2',
    style: {
      endArrow: {
        path: 'M -4,0 L 4,-4 L 4,4 Z', // 自定义箭头路径
        d: 0, // 偏移量
        fill: 'grey'
      }
    }
  }
});
graph.data(data);
graph.render();
let dataForGPU = {
  nodes: graph.cfg.data.nodes.map((d) => { if (d.size.length) { d.size = d.size[0] } return d }),
  edges: graph.cfg.data.edges
}
console.log("dataForGPU", dataForGPU);

initWebGPU(dataForGPU);




