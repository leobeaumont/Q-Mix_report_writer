import math
from utils.globals import ExecutionTrace
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Slider, Arrow, VeeHead, Div
from bokeh.layouts import column, row
from bokeh.io import output_file
from typing import List
import markdown

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
        
        # 2. Create the Info Panel (The right-hand side text display)
        self.info_panel = Div(
            text="<div style='font-family: sans-serif;'><h1>Step Details</h1><p>Move the slider to see agent interactions.</p></div>",
            width=450,
            styles={
                "background-color": "#fdfdfd",
                "padding": "20px",
                "border-left": "2px solid #eee",
                "height": "600px",
                "overflow-y": "auto",
                "box-shadow": "-2px 0 5px rgba(0,0,0,0.05)"
            }
        )

        # 3. Store ALL steps with round, execution info, and Dynamic HTML
        all_steps_data = {}
        for global_idx, (r_idx, num_executed) in enumerate(self.global_steps):
            n, e, rid = self._get_step_data(r_idx, num_executed)
            
            # Generate HTML description for this step
            current_round_data = self.trace[r_idx]
            exec_order = current_round_data.get("exec_order", [])
            active_agent = exec_order[num_executed - 1] if num_executed > 0 else "None (Start of Round)"
            

            action_val = None
            if num_executed > 0 and active_agent in current_round_data:
                action_val = current_round_data[active_agent].get('action', None)

            action_names = [
            "Solo process",  # 0
            "Broadcast",  #1
            "Selective query to LeadArchitect",  #2
            "Selective query to Researcher",  # 3
            "Selective query to DataAnalyst",  # 4
            "Selective query to TechnicalWriter",  # 5
            "Selective query to Reviewer",  # 6 
            "Aggregate",  # 7
            "Execute verify",  # 8
            "Debate with LeadArchitect",  # 9
            "Debate with Researcher",  # 10
            "Debate with DataAnalyst",  # 11
            "Debate with TechnicalWriter",  # 12
            "Debate with Reviewer",  # 13
            "Append",  # 14
            "Terminate"  # 15
            ]

            action_name = action_names[action_val] if action_val is not None else "None"

            step_html = f"""
                <div style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #34495e;">
                    <h2 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #f1c40f; padding-bottom: 10px;">
                        Round {r_idx} <small style="color: #bdc3c7; font-weight: normal;">(Step {global_idx})</small>
                    </h2>
                    <p><b>Agent Action:</b> 
                        <span style="background: #2c3e50; padding: 2px 8px; border-radius: 4px; color: #fff; font-weight: bold; font-family: monospace;">
                            { action_name }
                        </span>
                    </p>
                    <p style="font-size: 1.1em;"><b>Active Agent:</b> <code style="color: #e67e22;">{active_agent}</code></p>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
            """
            
            if num_executed > 0 and active_agent in current_round_data:
                agent_info = current_round_data[active_agent]
                
                # Defensive check: use 'or' to handle None values safely
                prompt_md = agent_info.get('prompt') or 'N/A'
                response_md = agent_info.get('response') or 'N/A'
                
                prompt_html = markdown.markdown(prompt_md, extensions=['fenced_code', 'tables'])
                response_html = markdown.markdown(response_md, extensions=['fenced_code', 'tables'])
                
                step_html += f"""
                    <h3 style="color: #2980b9;">Agent Logic</h3>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #3498db; margin-bottom: 10px;">
                        <p><b>Prompt:</b><br><div style="color: #7f8c8d; font-size: 0.9em;">{prompt_html}</div></p>
                    </div>
                    <div style="background: #fdf9ea; padding: 10px; border-radius: 5px; border-left: 4px solid #f1c40f;">
                        <p><b>Response:</b><br><div style="color: #7f8c8d; font-size: 0.9em;">{response_html}</div></p>
                    </div>
                """
            
            step_html += "</div>"

            all_steps_data[str(global_idx)] = {
                'nodes': n, 
                'edges': e, 
                'round_id': rid,
                'num_executed': num_executed,
                'step_text': step_html
            }

        self.plot = figure(
            title="Agent Communication Trace",
            x_range=(-radius-50, radius+50), y_range=(-radius-50, radius+50),
            tools="pan,wheel_zoom,reset,save", toolbar_location="above", match_aspect=True,
            width=600, height=700,
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
        ]))

        # Updated CustomJS to handle the side-panel text update
        callback = CustomJS(args=dict(n_src=self.node_source, e_src=self.edge_source, info=self.info_panel, all_data=all_steps_data), code="""
            const step = Math.round(cb_obj.value).toString();
            const targetData = all_data[step];
            
            // 1. Update the Right Panel Text Immediately
            info.text = targetData['step_text'];
            
            // 2. Animation & Graph Update Logic
            if (window.currentRound === undefined) window.currentRound = 0;
            const roundChanged = window.currentRound !== targetData['round_id'];
            window.currentRound = targetData['round_id'];

            if (window.bokehAnimationID) {
                cancelAnimationFrame(window.bokehAnimationID);
            }

            if (targetData['num_executed'] === 0 || roundChanged || 
                e_src.data['alpha'].length !== targetData['edges']['alpha'].length) {
                
                n_src.data = Object.assign({}, targetData['nodes']);
                e_src.data = Object.assign({}, targetData['edges']);
                n_src.change.emit();
                e_src.change.emit();
                return; 
            }

            const duration = 600;
            const startTime = performance.now();
            
            const startNodeAlpha = [...n_src.data['alpha']];
            const startEdgeAlpha = [...e_src.data['alpha']];
            const targetNodeAlpha = targetData['nodes']['alpha'];
            const targetEdgeAlpha = targetData['edges']['alpha'];

            const intermediateEdges = Object.assign({}, targetData['edges']);
            intermediateEdges['alpha'] = startEdgeAlpha;
            e_src.data = intermediateEdges;
            
            const intermediateNodes = Object.assign({}, targetData['nodes']);
            intermediateNodes['alpha'] = startNodeAlpha;
            n_src.data = intermediateNodes;

            function animate(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);

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

        self.slider = Slider(start=0, end=len(self.global_steps) - 1, value=0, step=1, title="Execution Step", sizing_mode="stretch_width")
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

                    is_visible = agent_key in all_executed_indices
                    e_data["alpha"].append(0.8 if is_visible else 0.0)
        
        return n_data, e_data, round_idx

    def show(self):
        output_file("agent_trace.html")
        # Layout: Slider on top, Plot and Info side-by-side
        final_layout = column(
            self.slider, 
            row(self.plot, self.info_panel, sizing_mode="stretch_both"),
            sizing_mode="stretch_both"
        )
        show(final_layout)

# --- EXECUTION ---
trace = ExecutionTrace.instance().load_trace()
viz = StandaloneVisualizer(trace)
viz.show()