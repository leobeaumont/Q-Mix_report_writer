import math
from utils.globals import ExecutionTrace
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Slider, Arrow, VeeHead, Div, Button
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

        self.style_fix = Div(text="""
            <style>
                body { 
                    margin: 0 !important; 
                    padding: 0 !important; 
                    overflow: hidden !important; 
                }
            </style>
        """)
        
        # 1. Setup Data Sources
        node_data, edge_data, _ = self._get_step_data(0, 0)
        self.node_source = ColumnDataSource(data=node_data)
        self.edge_source = ColumnDataSource(data=edge_data)
        
        # 2. Create the Panels
        self.info_panel = Div(
            text="<div style='font-family: sans-serif;'><h1>Step Details</h1></div>",
            width=440,
            styles={
                "background-color": "#fdfdfd",
                "padding": "20px",
                "border-left": "2px solid #f1c40f",
                "height": "87vh",
                "overflow-y": "auto"
            }
        )

        # NEW: The Report Panel
        self.report_panel = Div(
            text="<div style='font-family: sans-serif;'><h1>Current Report</h1><p>Waiting for Collector...</p></div>",
            width=440,
            styles={
                "background-color": "#ffffff",
                "padding": "20px",
                "border-left": "2px solid #2ecc71",
                "height": "87vh",
                "overflow-y": "auto",
                "box-shadow": "-2px 0 5px rgba(0,0,0,0.05)"
            }
        )

        # 3. Store ALL steps with round, execution info, and Dynamic HTML
        all_steps_data = {}
        latest_report_html = "<i>No report content generated yet.</i>"

        for global_idx, (r_idx, num_executed) in enumerate(self.global_steps):
            n, e, rid = self._get_step_data(r_idx, num_executed)
            
            current_round_data = self.trace[r_idx]
            exec_order = current_round_data.get("exec_order", [])
            
            # Identify which agent just acted
            last_agent_to_act = exec_order[num_executed - 1] if num_executed > 0 else None
            # Identify which agent is currently active/selected
            active_agent = last_agent_to_act if last_agent_to_act else "None (Start of Round)"
            
            # --- UPDATED REPORT LOGIC ---
            # Only update the stored report text IF the Collector was the one who just finished acting
            if last_agent_to_act == "Collector":
                collector_data = current_round_data.get("Collector", {})
                report_md = collector_data.get("report_state")
                if report_md:
                    latest_report_html = markdown.markdown(report_md, extensions=['fenced_code', 'tables'])

            # Wrap the report in a "bubble" style matching your other panels
            report_bubble_html = f"""
                <div style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
                    <h2 style="color: #2c3e50; border-bottom: 2px solid #2ecc71; padding-bottom: 10px;">Current Report State</h2>
                    <div style="background: #ebfaf0; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71; margin-top: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #2f3640;">
                        {latest_report_html}
                    </div>
                </div>
            """

            action_val = None
            if num_executed > 0 and last_agent_to_act in current_round_data:
                action_val = current_round_data[last_agent_to_act].get('action', None)

            action_names = [
                "Solo process", "Broadcast", "Selective query to LeadArchitect", 
                "Selective query to Researcher", "Selective query to DataAnalyst", 
                "Selective query to TechnicalWriter", "Selective query to Reviewer", 
                "Aggregate", "Execute verify", "Debate with LeadArchitect", 
                "Debate with Researcher", "Debate with DataAnalyst", 
                "Debate with TechnicalWriter", "Debate with Reviewer", 
                "Append", "Terminate"
            ]

            action_name = action_names[action_val] if (action_val is not None and action_val < len(action_names)) else "None"
            step_html = f"""
                <div style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #34495e;">
                    <h2 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #f1c40f; padding-bottom: 10px;">
                        Round {r_idx} <small style="color: #bdc3c7; font-weight: normal;">(Step {global_idx})</small>
                    </h2>
                    <p style="font-size: 1.1em;"><b>Agent Action:</b> 
                        <span style="background: #b08f0c; padding: 2px 8px; border-radius: 4px; color: #fff; font-weight: bold; font-family: monospace;">
                            { action_name }
                        </span>
                    </p>
                    <p style="font-size: 1.1em;"><b>Active Agent: <code style="color: #b08f0c;">{active_agent}</code></b></p>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
            """
            
            if num_executed > 0 and last_agent_to_act in current_round_data:
                agent_info = current_round_data[last_agent_to_act]
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
                'step_text': step_html,
                'report_text': report_bubble_html # Using the bubble variable here
            }

        self.plot = figure(
            title="Agent Communication Topology",
            x_range=(-radius-50, radius+50), y_range=(-radius-50, radius+50),
            tools="save", toolbar_location="above", match_aspect=True,
            width=500, height=800,
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

        # Updated CustomJS to include the report panel
        callback = CustomJS(args=dict(
            n_src=self.node_source, 
            e_src=self.edge_source, 
            info=self.info_panel, 
            report=self.report_panel, # Pass new panel
            all_data=all_steps_data
        ), code="""
            const step = Math.round(cb_obj.value).toString();
            const targetData = all_data[step];
            
            // Update the Panels
            info.text = targetData['step_text'];
            report.text = targetData['report_text']; // Update report text
            
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

        # 1. Update Slider to be narrower
        self.slider = Slider(start=0, end=len(self.global_steps) - 1, value=0, step=1,
                             width=1000, show_value=False, bar_color="#00000062")
        self.slider.js_on_change('value', callback)

        # 2. Define Control Buttons
        self.prev_btn = Button(label="◀ Previous", width=80, button_type="primary")
        self.next_btn = Button(label="Next ▶", width=80, button_type="primary")
        self.play_btn = Button(label="► Play", width=80, button_type="success")

        # 3. Add Button Logic via CustomJS
        self.prev_btn.js_on_click(CustomJS(args=dict(s=self.slider), code="""
            if (s.value > s.start) s.value -= 1;
        """))

        self.next_btn.js_on_click(CustomJS(args=dict(s=self.slider), code="""
            if (s.value < s.end) s.value += 1;
        """))

        # Play/Pause Logic
        self.play_btn.js_on_click(CustomJS(args=dict(s=self.slider, btn=self.play_btn), code="""
            if (window.playInterval) {
                clearInterval(window.playInterval);
                window.playInterval = null;
                btn.label = "► Play";
                btn.button_type = "success";
            } else {
                btn.label = "❚❚ Pause";
                btn.button_type = "danger";
                window.playInterval = setInterval(() => {
                    if (s.value < s.end) {
                        s.value += 1;
                    } else {
                        clearInterval(window.playInterval);
                        window.playInterval = null;
                        btn.label = "► Play";
                        btn.button_type = "success";
                    }
                }, 750); // 1 second per step
            }
        """))

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
        
        # Create the control bar row
        controls = row( 
            self.prev_btn, 
            self.play_btn, 
            self.next_btn, 
            self.slider,
            spacing=10, 
            align="center"
        )

        final_layout = column(
            self.style_fix,
            controls, # Use the new controls row instead of just self.slider
            row(self.plot, self.info_panel, self.report_panel, sizing_mode="stretch_height"),
            sizing_mode="stretch_both"
        )
        show(final_layout)

# --- EXECUTION ---
trace = ExecutionTrace.instance().load_trace()
viz = StandaloneVisualizer(trace)
viz.show()
