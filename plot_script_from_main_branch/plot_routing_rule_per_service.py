import pandas as pd
import os
from IPython.display import display
import graphviz

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def visualize_request_flow(fn, counter, plot_first_hop_only=False, include_legend=False):
    df = pd.read_csv(fn)
    if counter > 0:
        df = df[df['counter'] == counter]

    # Calculate the incoming RPS for `sslateingress`
    incoming_rps = df[df['src_svc'] == 'sslateingress']['flow'].sum()

    # Add an artificial "SOURCE" for incoming load only if `sslateingress` exists
    if 'sslateingress' in df['dst_svc'].values:
        dst_cid = df[df['dst_svc'] == 'sslateingress']['dst_cid'].iloc[0]
        artificial_rows = [{
            "src_svc": "SOURCE",
            "dst_svc": "sslateingress",
            "src_cid": "XXXX",
            "dst_cid": dst_cid,
            "flow": incoming_rps,
            "total": incoming_rps,
            "weight": 1.0
        }]
        artificial_df = pd.DataFrame(artificial_rows)
        df = pd.concat([df, artificial_df], ignore_index=True)
    else:
        print("No `sslateingress` found in the destination services.")

    # Debugging: Check if artificial rows are added
    print("DataFrame after adding SOURCE rows:")
    print(df[df['src_svc'] == 'SOURCE'])

    # Remove duplicates
    df = df.drop_duplicates(subset=["src_svc", "dst_svc", "src_cid", "dst_cid", "flow", "total", "weight"], keep='last')

    # Node color dictionary
    node_color_dict = {
        "us-west-1": "#FFBF00",
        "us-east-1": "#ff6375",
        "us-south-1": "#bbfbfc",
        "us-central-1": "#c8ffbf",
        "XXXX": "gray"
    }

    g_ = graphviz.Digraph()

    # Node and edge style parameters
    node_pw = "1.4"
    node_fs = "14"
    node_width = "0.6"
    edge_pw = "1.0"
    edge_fs = "12"
    edge_arrowsize = "1.0"
    edge_minlen = "1.0"
    fontname = "times bold italic"

    # Generate graph
    for index, row in df.iterrows():
        src_cid = row["src_cid"]
        dst_cid = row["dst_cid"]
        src_svc = row["src_svc"]
        dst_svc = row["dst_svc"]
        flow = row["flow"]
        weight = row["weight"]

        # Node labels
        src_node_label = f"{src_svc}\n{src_cid}" if src_svc != "SOURCE" else "SOURCE"
        dst_node_label = f"{dst_svc}\n{dst_cid}"

        # Node colors
        src_node_color = node_color_dict.get(src_cid, "gray")
        dst_node_color = node_color_dict.get(dst_cid, "gray")

        # Edge style
        edge_style = "dashed" if src_cid != dst_cid else "solid"
        edge_color = "black" if edge_style == "solid" else "purple"
        edge_label = f'{flow} ({int(weight * 100)}%)'

        # Add nodes
        g_.node(
            name=f"{src_svc}_{src_cid}",
            label=src_node_label,
            shape="circle",
            style="filled",
            fillcolor=src_node_color,
            penwidth=node_pw,
            fontsize=node_fs,
            fontname=fontname,
            fixedsize="True",
            width=node_width
        )
        g_.node(
            name=f"{dst_svc}_{dst_cid}",
            label=dst_node_label,
            shape="circle",
            style="filled",
            fillcolor=dst_node_color,
            penwidth=node_pw,
            fontsize=node_fs,
            fontname=fontname,
            fixedsize="True",
            width=node_width
        )

        # Add edge
        g_.edge(
            f"{src_svc}_{src_cid}",
            f"{dst_svc}_{dst_cid}",
            label=edge_label,
            penwidth=edge_pw,
            style=edge_style,
            fontsize=edge_fs,
            fontcolor=edge_color,
            color=edge_color,
            arrowsize=edge_arrowsize,
            minlen=edge_minlen
        )

    # Legend (Optional)
    if include_legend:
        with g_.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", fontsize="10")
            for region, color in node_color_dict.items():
                legend.node(region, label=region, shape="circle", style="filled", fillcolor=color, fontsize="10", width="0.3", height="0.3")

    return g_

def run(input_file):
    counter = -1
    g_ = visualize_request_flow(input_file, counter, plot_first_hop_only=False, include_legend=False)
    output_file = os.path.splitext(input_file)[0]
    print(f"** Saved routing visualization: {output_file}.pdf")
    g_.render(output_file)

import sys
if __name__ == "__main__":
    input_file = sys.argv[1]
    run(input_file)
