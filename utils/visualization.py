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
            for num_executed in range(len(exec_order) + 1):
                self.global_steps.append((r_idx, num_executed))
        
        # 1. Setup Data Sources
        node_data, edge_data, _ = self._get_step_data(0, 0)
        self.node_source = ColumnDataSource(data=node_data)
        self.edge_source = ColumnDataSource(data=edge_data)
        
        # 2. Store ALL steps with round and execution info
        all_steps_data = {}
        for global_idx, (r_idx, num_executed) in enumerate(self.global_steps):
            n, e, rid = self._get_step_data(r_idx, num_executed)
            all_steps_data[str(global_idx)] = {
                'nodes': n, 
                'edges': e, 
                'round_id': rid,
                'num_executed': num_executed  # Needed for snapping logic
            }

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
        node_renderer = self.plot.circle(x="x", y="y", radius=20, fill_color="color", 
                                         source=self.node_source, fill_alpha='alpha', line_alpha='alpha')

        self.plot.add_tools(HoverTool(renderers=[node_renderer], tooltips=[
            ("Agent", "@agent_id"),
            ("Prompt", "@prompt"),
            ("Response", "@response")
        ]))

        # Modified CustomJS to use atomic data updates and snapping
        callback = CustomJS(args=dict(n_src=self.node_source, e_src=self.edge_source, all_data=all_steps_data), code="""
            const step = Math.round(cb_obj.value).toString();
            const targetData = all_data[step];
            
            // Detect transitions
            if (window.currentRound === undefined) window.currentRound = 0;
            const roundChanged = window.currentRound !== targetData['round_id'];
            window.currentRound = targetData['round_id'];

            if (window.bokehAnimationID) {
                cancelAnimationFrame(window.bokehAnimationID);
            }

            // LOGIC FIX: If going to Step 0 or changing rounds, snap immediately and EXIT.
            // This prevents "Grey Ghost Edges" from being drawn during the coordinate shift.
            if (targetData['num_executed'] === 0 || roundChanged || 
                e_src.data['alpha'].length !== targetData['edges']['alpha'].length) {
                
                n_src.data = Object.assign({}, targetData['nodes']);
                e_src.data = Object.assign({}, targetData['edges']);
                n_src.change.emit();
                e_src.change.emit();
                return; 
            }

            const duration = 800; // Smoother duration
            const startTime = performance.now();
            
            // Capture starting state for interpolation
            const startNodeAlpha = [...n_src.data['alpha']];
            const startEdgeAlpha = [...e_src.data['alpha']];
            const targetNodeAlpha = targetData['nodes']['alpha'];
            const targetEdgeAlpha = targetData['edges']['alpha'];

            // Sync structural properties (colors, widths, coords) BEFORE animation starts
            // But keep the current alpha to avoid the flash.
            const intermediateEdges = Object.assign({}, targetData['edges']);
            intermediateEdges['alpha'] = startEdgeAlpha;
            e_src.data = intermediateEdges;
            
            const intermediateNodes = Object.assign({}, targetData['nodes']);
            intermediateNodes['alpha'] = startNodeAlpha;
            n_src.data = intermediateNodes;

            function animate(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);

                // Update Alpha values only
                const currentNAlpha = n_src.data['alpha'];
                for (let i = 0; i < startNodeAlpha.length; i++) {
                    currentNAlpha[i] = startNodeAlpha[i] + (targetNodeAlpha[i] - startNodeAlpha[i]) * progress;
                }

                const currentEAlpha = e_src.data['alpha'];
                for (let i = 0; i < startEdgeAlpha.length; i++) {
                    currentEAlpha[i] = startEdgeAlpha[i] + (targetEdgeAlpha[i] - startEdgeAlpha[i]) * progress;
                }

                n_src.change.emit();
                e_src.change.emit();

                if (progress < 1) {
                    window.bokehAnimationID = requestAnimationFrame(animate);
                }
            }
            window.bokehAnimationID = requestAnimationFrame(animate);
        """)

        self.slider = Slider(start=0, end=len(self.global_steps) - 1, value=0, step=1, title="Execution Step")
        self.slider.js_on_change('value', callback)

    def _calculate_coords(self, agent_names: List[str]):
        return {name: (self.radius * math.cos(2*math.pi*i/self.n_agents), 
                    self.radius * math.sin(2*math.pi*i/self.n_agents)) for i, name in enumerate(agent_names)}

    def _get_step_data(self, round_idx, num_executed):
        current_round = self.trace[round_idx]
        exec_order = current_round.get("exec_order", [])
        
        active_agent = exec_order[num_executed - 1] if num_executed > 0 else None
        finished_indices = set(exec_order[:num_executed - 1]) if num_executed > 0 else set()
        all_executed_indices = set(exec_order[:num_executed])
        
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
            
            if agent_key == active_agent:
                n_data["color"].append("#f1c40f") # Gold
                n_data["alpha"].append(1.0)
            elif agent_key in finished_indices:
                n_data["color"].append("#e7c53f") # Tarnished gold
                n_data["alpha"].append(0.4)
            else:
                n_data["color"].append("#d1d1d1") # Waiting Grey
                n_data["alpha"].append(0.4)

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

                    if active_agent and (agent_key == active_agent or target == active_agent):
                        e_data["color"].append("#f1c40f")
                        e_data["width"].append(4)
                    else:
                        e_data["color"].append("#888888")
                        e_data["width"].append(1.5)

                    # Logic: Only show edges if the agent has started its turn
                    is_visible = agent_key in all_executed_indices
                    e_data["alpha"].append(0.8 if is_visible else 0.0)
        
        return n_data, e_data, round_idx

    def show(self):
        output_file("agent_trace.html")
        show(column(self.slider, self.plot))

# --- EXECUTION ---
trace = ExecutionTrace.instance().load_trace()
viz = StandaloneVisualizer(trace)
viz.show()
