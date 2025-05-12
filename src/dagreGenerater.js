
import dagre from "@dagrejs/dagre"

export function dagreLayout({nodes,edges}=
    {nodes:[0,1,2],edges:[{source:0,target:1},{source:0,target:2}]}
) {
    var g = new dagre.graphlib.Graph();
    g.setGraph({rankdir:'LR'});
    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(function () { return {}; });

    // Add nodes to the graph. The first argument is the node id. The second is
    // metadata about the node. In this case we're going to add labels to each of
    // our nodes.
    // g.setNode("kspacey", { label: "Kevin Spacey", width: 144, height: 100 });
    // g.setEdge("kspacey", "swilliams");
    nodes.forEach(n => {
        g.setNode(n, { label: n, width: 50, height: 20 });
    })
    edges.forEach(e => {
        g.setEdge(e.source, e.target)
    })
    dagre.layout(g);
    let result = {
        nodes:[],
        edges:[]
    }
    g.nodes().forEach(function (v) {
        result.nodes.push(g.node(v))
        // console.log("Node " + v + ": " + JSON.stringify(g.node(v)));
    });
    g.edges().forEach(function (e) {
        result.edges.push(g.edge(e))
        // console.log("Edge " + e.v + " -> " + e.w + ": " + JSON.stringify(g.edge(e)));
    });
    return result
}

