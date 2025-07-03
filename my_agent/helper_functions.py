from pathlib import Path


def save_graph_image(graph):
    img_data = graph.get_graph().draw_mermaid_png()

    # Save it to a file
    output_path = Path("graph_output.png")
    with open(output_path, "wb") as f:
        f.write(img_data)

    print(f"Mermaid graph saved to {output_path.resolve()}")
