from torchview import draw_graph
import graphviz

graphviz.set_jupyter_format('png')
model_graph = draw_graph(
    vae, 
    input_size=(1,1,128,128), 
    expand_nested=False,
    hide_inner_tensors=True,
)

model_graph.visual_graph