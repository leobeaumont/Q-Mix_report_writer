import math
from utils.globals import ExecutionTrace
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Slider, Arrow, NormalHead
from bokeh.layouts import column
from bokeh.io import output_file
from typing import List

class StandaloneVisualizer:
    def __init__(self, trace, radius=200):
        self.trace = trace
        self.radius = radius

        all_keys = list(trace[0].keys())
        if "exec_order" in all_keys:
            all_keys.remove("exec_order")

        self.agent_names = sorted(all_keys)
        self.n_agents = len(self.agent_names)

        self.coords = self._calculate_coords(self.agent_names)

        self.global_steps = []
        for r_idx, r_data in enumerate(trace):
            exec_order = r_data.get("exec_order", [])
            # We add a step for the 'start of round' (0 executed) 
            # plus one step for each agent in the execution order.
            for num_executed in range(len(exec_order) + 1):
                self.global_steps.append((r_idx, num_executed))
        
        # 1. Setup Data Sources
        # We start by displaying Round 0
        node_data, edge_data = self._get_step_data(0, 0)
        self.node_source = ColumnDataSource(data=node_data)
        self.edge_source = ColumnDataSource(data=edge_data)
        
        # 2. Store ALL rounds in a hidden dictionary for JavaScript to access
        all_steps_data = {}
        for global_idx, (r_idx, num_executed) in enumerate(self.global_steps):
            n, e = self._get_step_data(r_idx, num_executed)
            all_steps_data[str(global_idx)] = {'nodes': n, 'edges': e}

        # 3. Create Figure
        self.plot = figure(
            title="Agent Communication Trace",
            x_range=(-radius-50, radius+50), y_range=(-radius-50, radius+50),
            tools="pan,wheel_zoom,reset,save", toolbar_location="above"
        )
        self.plot.axis.visible = False
        self.plot.grid.grid_line_color = None

        # 4. Add Glyphs
        arrow_head = NormalHead(fill_color="#888888", fill_alpha=0.6, line_color="#888888", size=10)
        arrows = Arrow(end=arrow_head, 
                       x_start='x_start', y_start='y_start', 
                       x_end='x_end', y_end='y_end', 
                       source=self.edge_source,
                       line_color="#888888", line_alpha=0.6, line_width=2)
        
        self.plot.add_layout(arrows)
        node_renderer = self.plot.circle(x="x", y="y", radius=20, fill_color="color", source=self.node_source)

        # 5. Add Hover
        self.plot.add_tools(HoverTool(renderers=[node_renderer], tooltips=[
            ("Agent", "@agent_id"),
            ("Prompt", "@prompt"),
            ("Response", "@response")
        ]))

        # This code runs in your BROWSER, not in Python.
         # Update the CustomJS to use the global_idx
        callback = CustomJS(args=dict(n_src=self.node_source, e_src=self.edge_source, all_data=all_steps_data), code="""
            const step = Math.round(cb_obj.value).toString();
            const data = all_data[step];
            
            n_src.data = data['nodes'];
            e_src.data = data['edges'];
            
            n_src.change.emit();
            e_src.change.emit();
        """)

        # 6. The JavaScript "Slide Show" Logic
        self.slider = Slider(
            start=0, 
            end=len(self.global_steps) - 1, 
            value=0, 
            step=1, 
            title="Execution Step"
        )
        self.slider.js_on_change('value', callback)

    def _calculate_coords(self, agent_names: List[str]):
        return {name: (self.radius * math.cos(2*math.pi*i/self.n_agents), 
                    self.radius * math.sin(2*math.pi*i/self.n_agents)) for i, name in enumerate(agent_names)}

    def _get_step_data(self, round_idx, num_executed):
        current_round = self.trace[round_idx]
        exec_order = current_round.get("exec_order", [])
        
        # Identify which agents have already executed in this step
        executed_indices = set(exec_order[:num_executed])
        
        n_data = {"x": [], "y": [], "agent_id": [], "prompt": [], "response": [], "color": []}
        e_data = {"x_start": [], "y_start": [], "x_end": [], "y_end": []}
        offset = 22 

        for i in range(self.n_agents):
            agent_key = self.agent_names[i]
            info = current_round.get(agent_key, {})
            x, y = self.coords[agent_key]
            
            n_data["x"].append(x); n_data["y"].append(y)
            n_data["agent_id"].append(agent_key)
            n_data["prompt"].append(info.get("prompt", "N/A"))
            n_data["response"].append(info.get("response", "N/A"))
            
            # Highlight logic: Blue if executed, Grey if waiting
            is_executed = agent_key in executed_indices
            n_data["color"].append("#1f77b4" if is_executed else "#d1d1d1")
            
            # EDGE FILTER: Only display edges originating from nodes that have executed
            if is_executed:
                for target in info.get("message_to", []):
                    tx, ty = self.coords[target]
                    dx, dy = tx - x, ty - y
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    if dist > 0:
                        ux, uy = dx / dist, dy / dist
                        e_data["x_start"].append(x + (ux * offset))
                        e_data["y_start"].append(y + (uy * offset))
                        e_data["x_end"].append(tx - (ux * offset))
                        e_data["y_end"].append(ty - (uy * offset))
        
        return n_data, e_data

    def show(self):
        output_file("agent_trace.html")
        show(column(self.slider, self.plot))

# --- EXECUTION ---
trace = ExecutionTrace.instance().load_trace()
viz = StandaloneVisualizer(trace)
viz.show()
