import math
from utils.globals import ExecutionTrace
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Slider, Arrow, VeeHead
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
            tools="pan,wheel_zoom,reset,save", toolbar_location="above", match_aspect=True
        )
        self.plot.axis.visible = False
        self.plot.grid.grid_line_color = None

        # 4. Add Glyphs
        arrow_head = VeeHead(fill_color="#000000", line_color="#000000", size=10, fill_alpha='alpha', line_alpha='alpha')
        arrows = Arrow(end=arrow_head, 
                       x_start='x_start', y_start='y_start', 
                       x_end='x_end', y_end='y_end', 
                       source=self.edge_source,
                       line_color='color', line_alpha='alpha', line_width='width')
        
        self.plot.add_layout(arrows)
        node_renderer = self.plot.circle(x="x", y="y", radius=20, fill_color="color", source=self.node_source, fill_alpha='alpha', line_alpha='alpha')

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
            const targetData = all_data[step];

            // 1. Cancel any existing animation to prevent UI locking
            if (window.bokehAnimationID) {
                cancelAnimationFrame(window.bokehAnimationID);
            }

            const duration = 400; 
            const startTime = performance.now();

            // 2. IMMEDIATE UPDATE: Synchronize coordinates and non-animatable properties
            // This prevents the "zoom/crash" by ensuring data lengths always match
            n_src.data['color'] = targetData['nodes']['color'];
            n_src.data['prompt'] = targetData['nodes']['prompt'];
            n_src.data['response'] = targetData['nodes']['response'];

            e_src.data['x_start'] = targetData['edges']['x_start'];
            e_src.data['y_start'] = targetData['edges']['y_start'];
            e_src.data['x_end'] = targetData['edges']['x_end'];
            e_src.data['y_end'] = targetData['edges']['y_end'];
            e_src.data['color'] = targetData['edges']['color'];
            e_src.data['width'] = targetData['edges']['width'];

            // Capture starting alphas for the transition
            const startNodeAlpha = [...n_src.data['alpha']];
            const startEdgeAlpha = [...e_src.data['alpha']];
            const targetNodeAlpha = targetData['nodes']['alpha'];
            const targetEdgeAlpha = targetData['edges']['alpha'];

            function animate(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);

                // Interpolate Node Alpha
                for (let i = 0; i < startNodeAlpha.length; i++) {
                    n_src.data['alpha'][i] = startNodeAlpha[i] + (targetNodeAlpha[i] - startNodeAlpha[i]) * progress;
                }

                // Interpolate Edge Alpha (Only if lengths match, otherwise snap)
                if (startEdgeAlpha.length === targetEdgeAlpha.length) {
                    for (let i = 0; i < startEdgeAlpha.length; i++) {
                        e_src.data['alpha'][i] = startEdgeAlpha[i] + (targetEdgeAlpha[i] - startEdgeAlpha[i]) * progress;
                    }
                } else {
                    e_src.data['alpha'] = targetEdgeAlpha;
                }

                n_src.change.emit();
                e_src.change.emit();

                if (progress < 1) {
                    window.bokehAnimationID = requestAnimationFrame(animate);
                }
            }

            window.bokehAnimationID = requestAnimationFrame(animate);
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
        active_agent = exec_order[num_executed - 1] if num_executed > 0 else None
        executed_indices = set(exec_order[:num_executed])
        
        n_data = {"x": [], "y": [], "agent_id": [], "prompt": [], "response": [], "color": [], "alpha": []}
        e_data = {"x_start": [], "y_start": [], "x_end": [], "y_end": [], "color": [], "width": [], "alpha": []}
        offset = 22 

        for i in range(self.n_agents):
            agent_key = self.agent_names[i]
            info = current_round.get(agent_key, {})
            x, y = self.coords[agent_key]
            
            n_data["x"].append(x); n_data["y"].append(y)
            n_data["agent_id"].append(agent_key)
            n_data["prompt"].append(info.get("prompt", "N/A"))
            n_data["response"].append(info.get("response", "N/A"))
            
            # Node Highlight logic: Gold for Active, Blue for Executed, Grey for Waiting
            if agent_key == active_agent:
                n_data["color"].append("#f1c40f") # Gold
            elif agent_key in executed_indices:
                n_data["color"].append("#d2b84d") # Tarnished gold
            else:
                n_data["color"].append("#d1d1d1") # Grey

            # 1.0 for active/executed, 0.3 for waiting
            n_alpha = 1.0 if (agent_key == active_agent) else 0.4
            n_data["alpha"].append(n_alpha)

            # Edge logic
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

                    # Edge Highlight logic: Red if connected to the active agent
                    if active_agent and (agent_key == active_agent or target == active_agent):
                        e_data["color"].append("#f1c40f") # Gold
                        e_data["width"].append(4)          # Thick
                    else:
                        e_data["color"].append("#888888") # Grey
                        e_data["width"].append(1.5)        # Normal

                    # If the source node hasn't executed yet, the edge is invisible
                    is_visible = agent_key in executed_indices
                    e_data["alpha"].append(0.8 if is_visible else 0.0)
        
        return n_data, e_data

    def show(self):
        output_file("agent_trace.html")
        show(column(self.slider, self.plot))

# --- EXECUTION ---
trace = ExecutionTrace.instance().load_trace()
viz = StandaloneVisualizer(trace)
viz.show()
