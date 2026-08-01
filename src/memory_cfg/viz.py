"""Figures for the CFG / FCG stages and for training results.

Everything here writes to a file by default. The original code called
`plt.show()`, which blocks forever on a headless box and produced nothing you
could put in a README.
"""
import os

import networkx as nx

# entry / exit / loop-header colouring, plus a neutral body colour
_ENTRY = "#2f9e44"
_EXIT = "#e03131"
_BODY = "#4c6ef5"
_LOOP = "#f08c00"
_API = "#ae3ec9"


def _plt():
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")   # headless: render to file, never to a window
    import matplotlib.pyplot as plt
    return plt


def _save_or_show(fig, save_to, plt):
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_to
    plt.show()
    return None


def layered_layout(cfg):
    """Top-down layout: control flow reads far better than a spring blob."""
    if cfg.number_of_nodes() == 0:
        return {}
    entries = [n for n in cfg.nodes if cfg.in_degree(n) == 0] or [min(cfg.nodes)]
    depth = {}
    frontier, level = list(entries), 0
    seen = set(entries)
    while frontier:
        for node in frontier:
            depth[node] = level
        nxt = []
        for node in frontier:
            for succ in cfg.successors(node):
                if succ not in seen:
                    seen.add(succ)
                    nxt.append(succ)
        frontier, level = nxt, level + 1
    for node in cfg.nodes:                      # unreachable blocks
        depth.setdefault(node, level)

    by_level = {}
    for node, d in depth.items():
        by_level.setdefault(d, []).append(node)

    # Order each level by the average position of its predecessors (the classic
    # barycenter heuristic). Sorting by address instead leaves the picture full
    # of crossings that have nothing to do with the control flow.
    pos, order = {}, {}
    for d in sorted(by_level):
        nodes = sorted(by_level[d])
        if d > 0:
            def barycenter(node):
                parents = [order[p] for p in cfg.predecessors(node) if p in order]
                return (sum(parents) / len(parents)) if parents else 0.0
            nodes.sort(key=lambda n: (barycenter(n), n))
        for i, node in enumerate(nodes):
            order[node] = float(i)
            pos[node] = (i - (len(nodes) - 1) / 2.0, -float(d))
    return pos


def _back_edges(cfg, depth):
    """Edges that close a loop: strictly shallower target, or a self-loop.

    Using `<=` here also flagged same-depth cross edges - the two arms of an
    if/else rejoining - which are not loops and made half of every diagram
    look like a cycle.
    """
    return {(u, v) for u, v in cfg.edges() if depth[v] < depth[u] or u == v}


def block_label(cfg, node, max_instructions=4):
    """A short mnemonic preview so a block is identifiable in the picture."""
    instructions = cfg.nodes[node].get("instructions") or []
    head = [ins[2] for ins in instructions[:max_instructions]]
    if len(instructions) > max_instructions:
        head.append("...")
    return f"0x{node:x}\n" + "\n".join(head)


def draw_cfg(cfg, title="", save_to=None, with_instructions=True,
             figsize=None, max_instructions=4):
    """Draw one function CFG with entry/exit/loop-header colouring."""
    plt = _plt()
    pos = layered_layout(cfg)
    n = max(1, cfg.number_of_nodes())
    figsize = figsize or (max(6, min(20, n * 1.5)), max(4, min(16, n * 0.9)))

    depth = {node: -pos[node][1] for node in cfg.nodes}
    back_edges = _back_edges(cfg, depth)
    loop_heads = {v for _u, v in back_edges}

    colors = []
    for node in cfg.nodes:
        if node in loop_heads:
            colors.append(_LOOP)
        elif cfg.in_degree(node) == 0:
            colors.append(_ENTRY)
        elif cfg.out_degree(node) == 0:
            colors.append(_EXIT)
        else:
            colors.append(_BODY)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(cfg, pos, node_color=colors, node_size=2600,
                           node_shape="s", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        cfg, pos, ax=ax, edge_color="#495057", arrows=True, arrowsize=14,
        width=1.2, node_size=2600,
        connectionstyle="arc3,rad=0.08")
    if back_edges:
        nx.draw_networkx_edges(
            cfg, pos, edgelist=sorted(back_edges), ax=ax, edge_color=_LOOP,
            style="dashed", arrows=True, arrowsize=14, width=1.6,
            node_size=2600, connectionstyle="arc3,rad=0.25")

    labels = ({node: block_label(cfg, node, max_instructions) for node in cfg.nodes}
              if with_instructions else {node: f"0x{node:x}" for node in cfg.nodes})
    nx.draw_networkx_labels(cfg, pos, labels, font_size=6.5,
                            font_color="white", ax=ax)
    ax.set_title(title or "Control-flow graph", fontsize=11)
    ax.axis("off")
    _legend(ax, plt, [(_ENTRY, "entry"), (_BODY, "block"), (_EXIT, "exit/return"),
                      (_LOOP, "loop header / back edge")])
    return _save_or_show(fig, save_to, plt)


def draw_cfg_grid(cfgs, save_to=None, title="", columns=3, max_functions=6):
    """A contact sheet of several function CFGs from one binary."""
    plt = _plt()
    chosen = sorted(cfgs.items(),
                    key=lambda kv: kv[1].number_of_nodes(),
                    reverse=True)[:max_functions]
    if not chosen:
        raise ValueError("no CFGs to draw")
    columns = min(columns, len(chosen))
    rows = (len(chosen) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.4 * rows))
    axes = [axes] if len(chosen) == 1 else list(axes.flatten())
    for ax, (offset, cfg) in zip(axes, chosen):
        pos = layered_layout(cfg)
        depth = {node: -pos[node][1] for node in cfg.nodes}
        colors = [_ENTRY if cfg.in_degree(n) == 0 else
                  _EXIT if cfg.out_degree(n) == 0 else _BODY for n in cfg.nodes]
        back = sorted(_back_edges(cfg, depth))
        nx.draw_networkx_nodes(cfg, pos, node_color=colors, node_size=420,
                               node_shape="s", alpha=0.9, ax=ax)
        nx.draw_networkx_edges(cfg, pos, ax=ax, edge_color="#868e96",
                               arrows=True, arrowsize=8, width=0.9,
                               node_size=420)
        if back:
            nx.draw_networkx_edges(cfg, pos, edgelist=back, ax=ax,
                                   edge_color=_LOOP, style="dashed",
                                   arrows=True, arrowsize=8, width=1.1,
                                   node_size=420,
                                   connectionstyle="arc3,rad=0.25")
        ax.set_title(f"0x{offset:x}  ({cfg.number_of_nodes()} blocks, "
                     f"{cfg.number_of_edges()} edges)", fontsize=9)
        ax.axis("off")
    for ax in axes[len(chosen):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return _save_or_show(fig, save_to, plt)


def draw_call_graph(fcg, save_to=None, title="", max_nodes=120, seed=7):
    """Draw the inter-function graph, with imports highlighted."""
    plt = _plt()
    graph = fcg
    if graph.number_of_nodes() > max_nodes:
        keep = sorted(graph.nodes,
                      key=lambda n: graph.degree(n), reverse=True)[:max_nodes]
        graph = graph.subgraph(keep)

    pos = nx.spring_layout(graph, seed=seed, k=0.7)
    is_api = nx.get_node_attributes(graph, "is_api")
    colors = [_API if is_api.get(n) else _BODY for n in graph.nodes]
    sizes = [90 + 40 * graph.degree(n) for n in graph.nodes]

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes,
                           alpha=0.85, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#adb5bd",
                           arrows=True, arrowsize=8, width=0.7, alpha=0.8)
    labels = {n: graph.nodes[n].get("api_name", "").split("!")[-1]
              for n in graph.nodes if is_api.get(n)}
    nx.draw_networkx_labels(graph, pos, labels, font_size=6.5,
                            font_color="#5f3dc4", ax=ax)
    ax.set_title(title or "Inter-function call graph", fontsize=12)
    ax.axis("off")
    _legend(ax, plt, [(_BODY, "function (pooled CFG)"), (_API, "imported symbol")])
    return _save_or_show(fig, save_to, plt)


def draw_training_curves(history, save_to=None, title="Stage-2 training"):
    plt = _plt()
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [h["loss"] for h in history], marker="o",
            color=_BODY, label="total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if "train_accuracy" in history[0]:
        twin = ax.twinx()
        twin.plot(epochs, [h["train_accuracy"] for h in history], marker="s",
                  color=_ENTRY, label="train accuracy")
        twin.set_ylabel("train accuracy")
        twin.set_ylim(0, 1.05)
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="center right")
    else:
        ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save_or_show(fig, save_to, plt)


def draw_confusion(matrix, labels, save_to=None, title="Confusion matrix",
                   cmap="Blues"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(labels),
                                    1.6 + 0.8 * len(labels)))
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    peak = max((max(row) for row in matrix), default=0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > peak / 2 else "#212529", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save_or_show(fig, save_to, plt)


def _legend(ax, plt, entries):
    handles = [plt.Line2D([0], [0], marker="s", color="none", label=label,
                          markerfacecolor=color, markersize=9)
               for color, label in entries]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
