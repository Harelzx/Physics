from nicegui import ui, events
from physics import VolleyballSimulation
import math
import plotly.graph_objects as go


# English text constants
ENGLISH_TEXT = {
    'title': 'Volleyball Trajectory Simulator',
    'h_label': 'Serve Height (h) [m]',
    'D_label': 'Distance from Serve Line (D) [m]',
    'v0_label': 'Initial Speed (v0) [m/s]',
    'alpha_label': 'Serve Angle (α) [deg]',
    'c_label': 'Drag Coefficient (c) [kg/m]',
    'm_label': 'Ball Mass (m) [kg]',
    'simulate_btn': 'Calculate Trajectory',
    'results_title': 'Results',
    'cleared_net': 'Cleared Net?',
    'in_bounds': 'Landed In Bounds?',
    'time_impact': 'Time to Impact',
    'max_height': 'Max Height',
    'time_max_height': 'Time to Max Height [t]',
    'time_return_h': 'Time to Return to Initial Height [t]',
    'g_label': 'Gravity (g) [m/s²]',
    'yes': 'Yes',
    'no': 'No',
    'na': 'N/A',
    'hit_net': 'Hit Net',
    'tab_sim': 'Simulation',
}

def main():
    #Results labels map
    result_labels = {}
    
    # UI Layout
    with ui.column().classes('w-full items-center p-2'):
        ui.label(ENGLISH_TEXT['title']).classes('text-2xl font-bold mb-2')
        
        # Container for plot
        plot_container = ui.plotly(go.Figure()).classes('w-full h-96')
        
        # Measured data state for overlay
        measured_data = {'x': [], 'y': [], 'loaded': False}
        
        with ui.row().classes('w-full gap-4 flex-wrap justify-center'):
            # Controls Column
            with ui.card().classes('min-w-[300px] flex-1 p-4'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['h_label']).classes('font-medium')
                    h_number = ui.number(value=2.5, min=0, step=0.01, format='%.2f').classes('w-24').props('dense')
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    h_input = ui.slider(min=1, max=4, step=0.01, value=2.5).props('label-always').classes('flex-grow')
                h_input.bind_value(h_number, 'value')

                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['D_label']).classes('font-medium')
                    D_number = ui.number(value=0, step=0.01, format='%.2f').classes('w-24').props('dense')
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    D_input = ui.slider(min=-2, max=4, step=0.01, value=0).props('label-always').classes('flex-grow')
                D_input.bind_value(D_number, 'value')

                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['v0_label']).classes('font-medium')
                    v0_number = ui.number(value=18, min=0, step=0.01, format='%.2f').classes('w-24').props('dense')
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    v0_input = ui.slider(min=5, max=30, step=0.01, value=18).props('label-always').classes('flex-grow')
                v0_input.bind_value(v0_number, 'value')

                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['alpha_label']).classes('font-medium')
                    alpha_number = ui.number(value=10, step=0.1, format='%.1f').classes('w-24').props('dense')
                with ui.row().classes('w-full items-center gap-2 mb-2'):
                    alpha_input = ui.slider(min=-90, max=90, step=0.1, value=10).props('label-always').classes('flex-grow')
                alpha_input.bind_value(alpha_number, 'value')

                alpha_input.bind_value(alpha_number, 'value')

                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['c_label']).classes('font-medium text-sm')
                    c_input = ui.number(value=0.005, min=0, step=0.0001, format='%.4f').classes('w-24').props('dense')

                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['m_label']).classes('font-medium text-sm')
                    m_input = ui.number(value=0.27, min=0.1, max=0.5, step=0.01, format='%.2f').classes('w-24').props('dense')

                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(ENGLISH_TEXT['g_label']).classes('font-medium text-sm')
                    g_input = ui.number(value=9.81, min=0, step=0.01, format='%.2f').classes('w-24').props('dense')
                
                sim_btn = ui.button(ENGLISH_TEXT['simulate_btn']).classes('mt-4 w-full bg-blue-500 text-white')
                

            # Results Column
            with ui.card().classes('min-w-[300px] flex-1 p-4'):
                ui.label(ENGLISH_TEXT['results_title']).classes('text-xl font-bold mb-2')
                result_labels['cleared_net'] = ui.label(f"{ENGLISH_TEXT['cleared_net']}: -")
                result_labels['in_bounds'] = ui.label(f"{ENGLISH_TEXT['in_bounds']}: -")
                result_labels['time_impact'] = ui.label(f"{ENGLISH_TEXT['time_impact']}: -")
                result_labels['max_height'] = ui.label(f"{ENGLISH_TEXT['max_height']}: -")
                result_labels['time_max_height'] = ui.label(f"{ENGLISH_TEXT['time_max_height']}: -")
                result_labels['time_return_h'] = ui.label(f"{ENGLISH_TEXT['time_return_h']}: -")

                # Load Measured Data Overlay
                ui.label('Overlay Measured Data').classes('mt-4 font-bold')
                
                # Overlay controls
                with ui.expansion('Overlay Settings', icon='settings').classes('w-full'):
                        def on_flip_x_change(_e):
                            # Recalculate X offset when flip changes so first point stays at x=-9
                            if measured_data['loaded'] and measured_data['x']:
                                first_x = measured_data['x'][0]
                                flip_x = -1 if overlay_flip_x.value else 1
                                overlay_off_x.value = -9.0 - (first_x * flip_x)
                            run_simulation()

                        overlay_flip_x = ui.checkbox('Flip X', value=True, on_change=on_flip_x_change)
                        overlay_off_x = ui.number(label='Offset X (m)', value=-9.0, step=0.01, format='%.4f', on_change=lambda e: run_simulation()).classes('w-full')
                        overlay_off_y = ui.number(label='Offset Y (m)', value=0.0, step=0.01, format='%.4f', on_change=lambda e: run_simulation()).classes('w-full')
                        overlay_status = ui.label('Status: No data').classes('text-xs text-gray-500 mt-1')
                        
                async def handle_overlay_upload(e: events.UploadEventArguments):
                    try:
                        content = await e.file.read()
                        csv_text = content.decode('utf-8')
                        lines = csv_text.strip().split('\n')
                        if len(lines) < 2: 
                            ui.notify('Empty CSV', type='warning')
                            return
                            
                        # Robust header parsing: detect delimiter (tab or comma)
                        first_line = lines[0].strip()
                        delimiter = '\t' if '\t' in first_line else ','

                        header = [h.strip() for h in first_line.split(delimiter)]

                        has_meters = 'x_meter' in header and 'y_meter' in header

                        xs, ys = [], []

                        # Indices for columns
                        try:
                            idx_x = header.index('x_meter') if has_meters else 2
                            idx_y = header.index('y_meter') if has_meters else 3
                        except ValueError:
                                # Fallback if standard indices 2/3 fail (unlikely given length check below)
                                idx_x, idx_y = 0, 1

                        # If fall back to pixels, check bounds
                        if not has_meters and len(header) < 4:
                                idx_x, idx_y = 2, 3

                        for line in lines[1:]:
                            parts = line.split(delimiter)
                            if len(parts) > max(idx_x, idx_y):
                                try:
                                    xs.append(float(parts[idx_x].strip()))
                                    ys.append(float(parts[idx_y].strip()))
                                except ValueError:
                                    continue
                                
                        if not xs: 
                            ui.notify('No valid data points found', type='warning')
                            return
                        
                        measured_data['x'] = xs
                        measured_data['y'] = ys
                        measured_data['loaded'] = True
                        measured_data['is_calibrated'] = has_meters
                        
                        if has_meters:
                            # Auto-set X offset so first point is at x = -9 (serve line)
                            # Account for flip_x: final_x = first_x * flip_x + off_x = -9
                            # So: off_x = -9 - first_x * flip_x
                            first_x = xs[0]
                            flip_x = -1 if overlay_flip_x.value else 1
                            overlay_off_x.value = -9.0 - (first_x * flip_x)
                            overlay_status.text = f'Status: Calibrated Data (Meters). Loaded {len(xs)} points.'
                            overlay_status.classes('text-green-600', remove='text-gray-500 text-orange-500')
                            ui.notify(f'Calibrated data loaded. X offset auto-set to {overlay_off_x.value:.2f}m', type='positive')
                        else:
                            overlay_status.text = f'Status: Raw Pixel Data. Loaded {len(xs)} points.'
                            overlay_status.classes('text-orange-500', remove='text-gray-500 text-green-600')
                            ui.notify(f'Loaded Pixel Data.', type='positive')

                        run_simulation() # Refresh plot
                        
                    except Exception as err:
                        ui.notify(f'Error: {err}', type='negative')


                # Hidden upload for overlay
                overlay_upload = ui.upload(on_upload=lambda e: handle_overlay_upload(e), auto_upload=True).props('accept=.csv').classes('hidden') 
                ui.button('Load CSV Overlay', on_click=lambda: overlay_upload.run_method('pickFiles')).classes('w-full bg-gray-600 text-white')

                # Fit Quality Section
                ui.separator().classes('my-2')
                ui.label("Fit Quality").classes('font-bold text-purple-600')
                fit_labels = {
                    'rmse': ui.label("RMSE: -").classes('text-sm'),
                    'r2': ui.label("R² Score: -").classes('text-sm'),
                    'max_err': ui.label("Max Error: -").classes('text-sm'),
                    'avg_err': ui.label("Avg Error: -").classes('text-sm'),
                    'status': ui.label("Run simulation with CSV loaded").classes('text-xs text-gray-500'),
                }


    # --- SIMULATION LOGIC ---
    def update_plot(sim_result):
        fig = go.Figure()

        # Unpack trajectory
        traj = sim_result['trajectory']
        x = traj['x']
        y = traj['y']
        
        # 1. Plot Simulated Trajectory
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Simulated',
            line=dict(color='blue', width=3)
        ))

        # 2. Overlay Measured Data
        if measured_data['loaded']:
            off_x = overlay_off_x.value if overlay_off_x.value is not None else 0.0
            off_y = overlay_off_y.value if overlay_off_y.value is not None else 0.0
            flip_x = -1 if overlay_flip_x.value else 1

            # Data is in meters - apply flip and offset
            xm = [p * flip_x + off_x for p in measured_data['x']]
            ym = [p + off_y for p in measured_data['y']]

            fig.add_trace(go.Scatter(
                x=xm, y=ym,
                mode='markers',
                name='Measured',
                marker=dict(color='green', size=6)
            ))

        # 3. Court Elements
        # Net
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, 2.43],
            mode='lines',
            name='Net (2.43m)',
            line=dict(color='red', width=4)
        ))
        
        # Court Lines (Serve -9m, End 9m)
        fig.add_trace(go.Scatter(
            x=[-9, -9], y=[0, 4],
            mode='lines',
            name='Serve Line',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[9, 9], y=[0, 4],
            mode='lines',
            name='End Line',
            line=dict(color='gray', width=2, dash='dash')
        ))

        # Serve Point (Start)
        if x:
            fig.add_trace(go.Scatter(
                x=[x[0]], y=[y[0]],
                mode='markers',
                name='Serve Point',
                marker=dict(color='black', size=8)
            ))

        # Layout Configuration
        fig.update_layout(
            title='Ball Trajectory (Interactive)',
            xaxis_title='Distance (m)',
            yaxis_title='Height (m)',
            xaxis=dict(range=[-10, 12], zeroline=True),
            yaxis=dict(range=[0, max(4.5, max(y) + 0.5) if y else 4.5], zeroline=True),
            width=None, # Auto width
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='closest'
        )

        plot_container.update_figure(fig)

    def run_simulation():
        h = h_input.value
        D = D_input.value
        v0 = v0_input.value
        alpha = alpha_input.value
        c = c_input.value
        m = m_input.value
        g = max(0.01, g_input.value) 
        if g_input.value < 0:
            ui.notify("Gravity must be positive. Using 0.01 as fallback.", type='warning')
        
        sim = VolleyballSimulation(h, D, v0, alpha, c, m=m, g=g)
        res = sim.simulate()
        
        # Update Results
        result_labels['cleared_net'].text = f"{ENGLISH_TEXT['cleared_net']}: {ENGLISH_TEXT['yes'] if res['cleared_net'] else ENGLISH_TEXT['no']}"
        if res['hit_net']:
             result_labels['cleared_net'].text += f" ({ENGLISH_TEXT['hit_net']})"
             
        result_labels['in_bounds'].text = f"{ENGLISH_TEXT['in_bounds']}: {ENGLISH_TEXT['yes'] if res['in_bounds'] else ENGLISH_TEXT['no']}"
        result_labels['time_impact'].text = f"{ENGLISH_TEXT['time_impact']}: {res['t_final']:.2f} s"
        result_labels['max_height'].text = f"{ENGLISH_TEXT['max_height']}: {res['max_height']:.2f} m"
        result_labels['time_max_height'].text = f"{ENGLISH_TEXT['time_max_height']}: {res['t_max_height']:.2f} s"
        t_ret = res['t_peak_to_return']
        val_ret = f"{t_ret:.2f} s" if t_ret is not None else ENGLISH_TEXT['na']
        result_labels['time_return_h'].text = f"{ENGLISH_TEXT['time_return_h']}: {val_ret}"
        
        update_plot(res)

        # Update Fit Quality if measured data loaded
        if measured_data['loaded']:
            off_x = overlay_off_x.value if overlay_off_x.value is not None else 0.0
            off_y = overlay_off_y.value if overlay_off_y.value is not None else 0.0
            flip_x = -1 if overlay_flip_x.value else 1

            # Data is in meters - apply flip and offset
            xm = [p * flip_x + off_x for p in measured_data['x']]
            ym = [p + off_y for p in measured_data['y']]
            
            if ym:
                # Calculate Fit Quality metrics
                # For each measured point, find the closest simulated Y at that X
                sim_x = res['trajectory']['x']
                sim_y = res['trajectory']['y']

                errors = []

                for mx, my in zip(xm, ym):
                    # Find the simulated Y value at measured X by interpolation
                    # Find bracketing indices in sim_x
                    sim_y_at_mx = None
                    for i in range(len(sim_x) - 1):
                        if (sim_x[i] <= mx <= sim_x[i+1]) or (sim_x[i] >= mx >= sim_x[i+1]):
                            # Linear interpolation
                            if sim_x[i+1] != sim_x[i]:
                                t = (mx - sim_x[i]) / (sim_x[i+1] - sim_x[i])
                                sim_y_at_mx = sim_y[i] + t * (sim_y[i+1] - sim_y[i])
                            else:
                                sim_y_at_mx = sim_y[i]
                            break

                    if sim_y_at_mx is not None:
                        err_y = my - sim_y_at_mx
                        errors.append(err_y ** 2)

                if errors:
                    # RMSE
                    rmse = math.sqrt(sum(errors) / len(errors))

                    # R² Score (coefficient of determination)
                    # R² = 1 - SS_res / SS_tot
                    mean_y = sum(ym) / len(ym)
                    ss_tot = sum((y - mean_y) ** 2 for y in ym)
                    ss_res = sum(errors)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    # Max and Average absolute error
                    abs_errors = [math.sqrt(e) for e in errors]
                    max_err = max(abs_errors)
                    avg_err = sum(abs_errors) / len(abs_errors)

                    # Update fit labels
                    fit_labels['rmse'].text = f"RMSE: {rmse:.4f} m"
                    fit_labels['r2'].text = f"R² Score: {r2:.4f}"
                    fit_labels['max_err'].text = f"Max Error: {max_err:.4f} m"
                    fit_labels['avg_err'].text = f"Avg Error: {avg_err:.4f} m"

                    # Color code based on quality
                    if r2 >= 0.98:
                        quality = "Excellent"
                        color = "text-green-600"
                    elif r2 >= 0.90:
                        quality = "Good"
                        color = "text-blue-600"
                    elif r2 >= 0.80:
                        quality = "Fair"
                        color = "text-yellow-600"
                    else:
                        quality = "Poor"
                        color = "text-red-600"

                    fit_labels['status'].text = f"Quality: {quality} (R²: {r2:.2f}) | Points Compared: {len(errors)}/{len(xm)}"
                    fit_labels['status'].classes(color, remove='text-gray-500 text-green-600 text-blue-600 text-yellow-600 text-red-600')
                else:
                    fit_labels['status'].text = "No common X-range between data and simulation"
                    fit_labels['status'].classes('text-orange-500', remove='text-gray-500 text-green-600 text-blue-600 text-yellow-600 text-red-600')

        else:
             # Reset fit labels
             fit_labels['rmse'].text = "RMSE: -"
             fit_labels['r2'].text = "R² Score: -"
             fit_labels['max_err'].text = "Max Error: -"
             fit_labels['avg_err'].text = "Avg Error: -"
             fit_labels['status'].text = "Load CSV data first"
             fit_labels['status'].classes('text-gray-500', remove='text-green-600 text-blue-600 text-yellow-600 text-red-600 text-orange-500')

    sim_btn.on_click(run_simulation)
    run_simulation() # Run initially on startup
    
    ui.run(title=ENGLISH_TEXT['title'])

if __name__ in {"__main__", "__mp_main__"}:
    main()