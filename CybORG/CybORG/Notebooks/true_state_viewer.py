from IPython.display import display
import graphviz
import numpy as np
import ipywidgets as widgets

class TrueStateTreeGraphViz:
    def __init__(self, true_state_numpy):
        g = graphviz.Digraph(format='png')
        g.attr(rankdir='TB')
        
        self.subnet_indices = [[8,9,10,11,12],[0,1,2,3],[4,5,6,7]]
        self.server_indices = [1,2,3,7]
        for i,indices in enumerate(self.subnet_indices):
            with g.subgraph() as s:
                sname = f"S{i+1}"
                s.attr(rank="same")
                s.node(sname, shape='Msquare')
                for i in indices:
                    node = self.get_node(s, sname,i,true_state_numpy)
        
        g.edge("S1","S2",arrowhead="none", penwidth="3")
        g.edge("S2","S3",arrowhead="none", penwidth="3")
        self.g = g
        
    def display(self):
        display(self.g)
        
    def get_node(self, subgraph, subnet_name, i, np_arr):
        name = self.index_to_name(i)
        subnet = subnet_name #self.index_to_subnet(i)
        known = bool(np_arr[i,0,1])
        scanned = bool(np_arr[i,0,2])
        user = bool(np_arr[i,1,1])
        priv = bool(np_arr[i,1,2])
        style = "dotted,filled"
        if known:
            style = "solid,filled"
        if scanned:
            style = "bold,filled"
        
        colour = "white"
        if user:
            colour = "lightgray"
        if priv:
            colour = "orange"
        
        shape = "circle"#"circle"if i not in self.server_indices else "doublecircle"
        
#         with subgraph:
        node = subgraph.node(name, shape=shape, fillcolor=colour, style=style)
        subgraph.edge(name,subnet,arrowhead="none")
        return node
        
    def index_to_name(self, index):
        if index==0:
            return "D"
        if index <4:
            return f"E{index}"
        if index <7:
            return f"O{index-4}"
        if index ==7:
            return "OS"
        else:
            return f"U{index-8}"


class BlueObsTreeGraphViz:
    def __init__(self, blue_obs_numpy):
        g = graphviz.Digraph(format='png')
        g.attr(rankdir='TB')
        
        self.subnet_indices = [[8,9,10,11,12],[0,1,2,3],[4,5,6,7]]
        self.server_indices = [1,2,3,7]
        for i,indices in enumerate(self.subnet_indices):
            with g.subgraph() as s:
                sname = f"S{i+1}"
                s.attr(rank="same")
                s.node(sname, shape='Msquare', color="blue", fontcolor="blue")
                for i in indices:
                    node = self.get_node(s, sname, i, blue_obs_numpy)
        
        g.edge("S1","S2",arrowhead="none", penwidth="3", color="blue")
        g.edge("S2","S3",arrowhead="none", penwidth="3", color="blue")
        self.g = g
        
    def display(self):
        display(self.g)
        
    def get_node(self, subgraph, subnet_name, i, np_arr):
        name = self.index_to_name(i)
        subnet = subnet_name #self.index_to_subnet(i)
        
        activity = np_arr[i,0]
        compromised = np_arr[i,1]
        # known = bool(np_arr[i,0,1])
        # scanned = bool(np_arr[i,0,2])
        
        # user = bool(np_arr[i,1,1])
        # priv = bool(np_arr[i,1,2])
        
        
        if activity == 2:
            # "None"
            style = "dotted,filled"
        elif activity == 1:
            # "Scan"
            style = "solid,filled"
        elif activity == 0:
            # "Exploit"
            style = "bold,filled"
        else:
            raise Exception(f"activity value not supported: {activity}")
        
        if compromised == 3:
            # "No"
            colour = "white"
        elif compromised == 2:
            # "Unknown"
            colour = "yellow"
        elif compromised == 1:
            # "User"
            colour = "lightgray"
        elif compromised == 0:
            # "Privileged"
            colour = "orange"
        else:
            raise Exception(f"compromised value not supported: {compromised}")
        
        shape = "doublecircle" if activity == 0 else "circle"
        
#         with subgraph:
        node = subgraph.node(name, shape=shape, fillcolor=colour, style=style, color="blue", fontcolor="blue")
        subgraph.edge(name,subnet,arrowhead="none", color="blue", fontcolor="blue")
        return node
        
    def index_to_name(self, index):
        if index==0:
            return "D"
        if index <4:
            return f"E{index}"
        if index <7:
            return f"O{index-4}"
        if index ==7:
            return "OS"
        else:
            return f"U{index-8}"

def display_tree_pairs(tree_pairs):
    length = len(tree_pairs)
    print(length)
    index = 0

    prev_button = widgets.Button(description="<<")
    next_button = widgets.Button(description=">>")
    int_slider = widgets.IntSlider(name='Index', min=0, max=length-1, step=1, value=index)
    # output = widgets.Output()
    sidebyside_controls = widgets.HBox([prev_button, next_button, int_slider])
    
    wi1 = widgets.Image(value = tree_pairs[index][0].g.pipe(), format='png', width=400)
    wi2 = widgets.Image(value = tree_pairs[index][1].g.pipe(), format='png', width=400)
    sidebyside_images = widgets.HBox([wi1, wi2])

    def update_images(index):
        nonlocal wi1, wi2, tree_pairs, sidebyside_controls, sidebyside_images
        wi1.value = tree_pairs[index][0].g.pipe()
        wi2.value = tree_pairs[index][1].g.pipe()
#         display(sidebyside_controls, sidebyside_images)
    
    def on_button_clicked(b):
        nonlocal index, length
        if b ==prev_button:
            if index >0:
                index -= 1
                update_images(index)
        else:
            if index < length-1:
                index += 1
                update_images(index)
        int_slider.value=index
    #     with output:


    def on_value_change(change):
        nonlocal index
        if index != change.new:
            index = change.new
            update_images(index)

    int_slider.observe(on_value_change, names='value')

    prev_button.on_click(on_button_clicked)
    next_button.on_click(on_button_clicked)
    # update_images(index)
    display(sidebyside_controls, sidebyside_images)

def display_tree_red_preds(tree_red_preds, pred_display_cap=None):
    length = len(tree_red_preds) 
    cap = pred_display_cap if pred_display_cap else len(tree_red_preds[0][3])
    print(length)
    index = 0

    prev_button = widgets.Button(description="<<")
    next_button = widgets.Button(description=">>")
    int_slider = widgets.IntSlider(name='Index', min=0, max=length-1, step=1, value=index)

    # output = widgets.Output()
    sidebyside_controls = widgets.HBox([prev_button, next_button, int_slider])
    
    im_pre = widgets.Image(value = tree_red_preds[index][0].g.pipe(), format='png', width=200)
    im_blue = widgets.Image(value = tree_red_preds[index][1].g.pipe(), format='png', width=200)
    im_red_true = widgets.Image(value = tree_red_preds[index][2].g.pipe(), format='png', width=200)
    sidebyside_truth_images = widgets.HBox([im_pre, im_blue, im_red_true])

    im_preds = []
    for tree_pred in tree_red_preds[index][3][:cap]:
        im_preds.append(widgets.Image(value = tree_pred.g.pipe(), format='png', width=150))
    sidebyside_pred_images = widgets.HBox(im_preds, overflow_x='auto', overflow='scroll')

    def update_images(index):
        nonlocal im_pre, im_blue, im_red_true, tree_red_preds, sidebyside_controls, sidebyside_truth_images, sidebyside_pred_images, cap
        im_pre.value = tree_red_preds[index][0].g.pipe()
        im_blue.value = tree_red_preds[index][1].g.pipe()
        im_red_true.value = tree_red_preds[index][2].g.pipe()
        for i, tree_pred in enumerate(tree_red_preds[index][3][:cap]):
            im_preds[i].value = tree_pred.g.pipe()
#       display(sidebyside_controls, sidebyside_images)
    
    def on_button_clicked(b):
        nonlocal index, length
        if b ==prev_button:
            if index >0:
                index -= 1
                update_images(index)
        else:
            if index < length-1:
                index += 1
                update_images(index)
        int_slider.value=index
    #     with output:


    def on_value_change(change):
        nonlocal index
        if index != change.new:
            index = change.new
            update_images(index)

    int_slider.observe(on_value_change, names='value')

    prev_button.on_click(on_button_clicked)
    next_button.on_click(on_button_clicked)
    
#     update_images(index)

    display(sidebyside_controls, sidebyside_truth_images, sidebyside_pred_images)