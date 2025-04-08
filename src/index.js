import { initWebGPU } from "./gpuRenderer";
import { dagreLayout } from "./dagreGenerater";



import './style.css';
import * as G6 from '@antv/g6';
import originData from '../data/data-11W.csv';
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
        return { source, target
            // , label 
        };
    });

    return edges;
}
let cardID = mergeAndDeduplicate(dat)
let eList = generateEdges(dat)
console.log(cardID);

console.log(eList);
// let dagreGraph = dagreLayout({nodes:cardID.splice(0,100), edges:[]})
// console.log(dagreGraph.node);

let data = {
    nodes:cardID.map(d => {
        return {
            id:d,
            // label:d
        }
    }),
    edges:eList,
}
// d3.csv("./data/data-11W.csv").then(function(data) {
//     // 2. 数据加载成功后的回调函数
//     console.log("原始数据：", data);

//     // 3. 处理数据（例如，将字符串转换为数字）
//     data.forEach(function(d) {
//       d.value = +d.value; // 将 value 列转换为数字
//     });

//     // 4. 打印处理后的数据
//     console.log("处理后的数据：", data);

//     // 5. 使用数据（例如，绘制图表）
//     // 这里可以调用 D3 的绘图方法
//   }).catch(function(error) {
//     // 6. 处理错误
//     console.error("加载 CSV 文件时出错：", error);
//   });


const graph = new G6.Graph({
    container: 'container',
    width:1000,
    height:1000,
    // fitView: true,
    modes: {
      default: [ 'drag-canvas', 'zoom-canvas' ]//, 'drag-node'
    },
    layout: {
      type: 'dagre',
      rankdir: 'LR',
      align: 'DL',
      nodesepFunc: () => {
        return 1;
      },
      ranksepFunc: () => {
        return 1;
      }
    },
    animate: true,
    defaultNode: {
      size: [ 50, 20 ],
        type: 'rect',
      style: {
        lineWidth: 2,
        stroke: '#5B8FF9',
        fill: '#C6E5FF',
        fillOpacity:0.1
      }
    },
    defaultEdge: {
    type:'cubic-horizontal',
      size: 1,
      color: '#e2e2e2',
      style: {
        endArrow: {
            path: 'M -4,0 L 4,-4 L 4,4 Z', // 自定义箭头路径
            d: 0, // 偏移量
            fill:'grey'
            }
      }
    }
  });
  graph.data(data);
  graph.render();
  initWebGPU(graph);
  console.log(graph)





