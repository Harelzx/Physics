from nicegui import ui, events
from physics import VolleyballSimulation
from video_analyzer import VideoAnalyzer
import tempfile
import os
import cv2
import base64
import asyncio
import traceback
import functools
import math
import plotly.graph_objects as go

def safe_handler(func):
    """Decorator to catch and display exceptions without crashing the app"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            ui.notify(error_msg, type='negative', timeout=5000)
    return wrapper

def safe_async_handler(func):
    """Decorator to catch and display exceptions in async functions without crashing the app"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            ui.notify(error_msg, type='negative', timeout=5000)
    return wrapper

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
    'time_max_height': 'Time to Max Height',
    'time_return_h': 'Time from Peak to Initial Height',
    'yes': 'Yes',
    'no': 'No',
    'na': 'N/A',
    'hit_net': 'Hit Net',
    'tab_sim': 'Simulation',
    'tab_video': 'Video Analysis',
    'upload_label': 'Upload Video',
    'calib_dist_label': 'Real Distance (m)',
    'calib_step1': 'Step 1: Set Scale',
    'calib_step2': 'Step 2: Set Origin',
}

def main():
    # State for Video Analysis
    analyzer = VideoAnalyzer()
    current_video_name = {'name': None}  # Store original video filename
    
    # Results labels map
    result_labels = {}
    
    # UI Layout
    with ui.column().classes('w-full items-center p-2'):
        ui.label(ENGLISH_TEXT['title']).classes('text-2xl font-bold mb-2')
        
        with ui.tabs().classes('w-full') as tabs:
            sim_tab = ui.tab(ENGLISH_TEXT['tab_sim'])
            video_tab = ui.tab(ENGLISH_TEXT['tab_video'])
            
        with ui.tab_panels(tabs, value=sim_tab).classes('w-full'):
            
            # --- SIMULATION TAB ---
            with ui.tab_panel(sim_tab):
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

                        with ui.row().classes('w-full items-center justify-between'):
                            ui.label(ENGLISH_TEXT['c_label']).classes('font-medium')
                            c_number = ui.number(value=0.005, min=0, step=0.0001, format='%.4f').classes('w-24').props('dense')
                        with ui.row().classes('w-full items-center gap-2 mb-2'):
                            c_input = ui.slider(min=0.001, max=0.01, step=0.0001, value=0.005).props('label-always').classes('flex-grow')
                        c_input.bind_value(c_number, 'value')

                        with ui.row().classes('w-full items-center justify-between'):
                            ui.label(ENGLISH_TEXT['m_label']).classes('font-medium')
                            m_number = ui.number(value=0.27, min=0.1, max=0.5, step=0.01, format='%.2f').classes('w-24').props('dense')
                        with ui.row().classes('w-full items-center gap-2'):
                            m_input = ui.slider(min=0.2, max=0.4, step=0.01, value=0.27).props('label-always').classes('flex-grow')
                        m_input.bind_value(m_number, 'value')
                        
                        sim_btn = ui.button(ENGLISH_TEXT['simulate_btn']).classes('mt-4 w-full bg-blue-500 text-white')
                        
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
                                print(f"DEBUG: CSV Header detected (delimiter={'tab' if delimiter == chr(9) else 'comma'}): {header}")

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
                                print(f"Error loading overlay: {err}")


                        # Hidden upload for overlay
                        overlay_upload = ui.upload(on_upload=lambda e: handle_overlay_upload(e), auto_upload=True).props('accept=.csv').classes('hidden') 
                        ui.button('Load CSV Overlay', on_click=lambda: overlay_upload.run_method('pickFiles')).classes('w-full bg-gray-600 text-white')


                    # Results Column
                    with ui.card().classes('min-w-[300px] flex-1 p-4'):
                        ui.label(ENGLISH_TEXT['results_title']).classes('text-xl font-bold mb-2')
                        result_labels['cleared_net'] = ui.label(f"{ENGLISH_TEXT['cleared_net']}: -")
                        result_labels['in_bounds'] = ui.label(f"{ENGLISH_TEXT['in_bounds']}: -")
                        result_labels['time_impact'] = ui.label(f"{ENGLISH_TEXT['time_impact']}: -")
                        result_labels['max_height'] = ui.label(f"{ENGLISH_TEXT['max_height']}: -")
                        result_labels['time_max_height'] = ui.label(f"{ENGLISH_TEXT['time_max_height']}: -")
                        result_labels['time_return_h'] = ui.label(f"{ENGLISH_TEXT['time_return_h']}: -")

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

            # --- VIDEO ANALYSIS TAB ---
            with ui.tab_panel(video_tab):
                with ui.row().classes('w-full gap-2'):
                    # Left Column: Controls & Status (narrower)
                    with ui.column().classes('w-1/4'):
                        ui.label(ENGLISH_TEXT['upload_label']).classes('font-bold')
                        ui.upload(on_upload=lambda e: handle_upload(e), auto_upload=True).classes('w-full')
                        
                        status_label = ui.label('Status: Waiting for video')

                        # Calibration Controls (simplified)
                        calib_card = ui.card().classes('w-full p-2 mt-2 hidden')
                        with calib_card:
                            ui.label('Calibration: Ball Diameter').classes('font-bold')
                            ui.label('Step 1: Click opposite edges of the ball').classes('text-sm text-gray-600')
                            
                            real_dist_input = ui.number(value=0.21, min=0.1, step=0.01, label='Ball Diameter (m)').classes('w-full')
                            
                            calib_btn = ui.button('Set Scale from Clicks', on_click=lambda: set_scale()).classes('w-full mt-2')
                            calib_btn.disable()
                            
                            ui.label('Step 2: Set Origin (0,0)').classes('mt-2 text-sm text-gray-600')
                            origin_btn = ui.button('Set Origin', on_click=lambda: enable_origin_mode()).classes('w-full mt-2')
                            origin_btn.disable()

                    # Right Column: Interactive Image & Plot (wider)
                    with ui.column().classes('w-3/4'):
                        # Interactive Image for clicks - larger
                        ii_container = ui.card().classes('w-full')
                        with ii_container:
                            interactive_image = ui.interactive_image(cross=True).classes('w-full').style('min-height: 500px')

                        # Unified Frame Editor (combines browsing and tracking correction)
                        frame_editor_card = ui.card().classes('w-full mt-2 hidden')
                        with frame_editor_card:
                            ui.label('Frame Editor - Click on ball to mark/correct position').classes('font-bold')

                            # Frame slider on its own row
                            with ui.row().classes('w-full items-center gap-2'):
                                frame_slider = ui.slider(min=0, max=100, step=1, value=0).classes('flex-grow')
                                frame_input = ui.number(min=0, max=100, step=1, value=0).classes('w-20').props('dense')
                            frame_label = ui.label('Frame: 0 / 0')
                            correction_status = ui.label('Click on ball to mark position').classes('text-xs text-gray-500')

                            # Navigation controls + Zoom all on one row
                            with ui.row().classes('w-full items-center gap-2 mt-2'):
                                prev_frame_btn = ui.button('◀').classes('bg-gray-500 text-white')
                                next_frame_btn = ui.button('▶').classes('bg-gray-500 text-white')
                                play_btn = ui.button('▶ Play').classes('bg-blue-500')
                                ui.label('|').classes('text-gray-400')
                                ui.label('Zoom:').classes('text-sm')
                                zoom_slider = ui.slider(min=1, max=4, step=0.5, value=1).props('label-always').classes('w-24')
                                zoom_label = ui.label('1x').classes('text-sm')

                            # Action buttons + progress
                            with ui.row().classes('w-full items-center gap-2 mt-2'):
                                clear_btn = ui.button('Clear All').classes('bg-red-500 text-white')
                                finish_btn = ui.button('Finish & Save').classes('bg-purple-600 text-white')
                                load_csv_btn = ui.button('Load CSV').classes('bg-gray-600 text-white')
                                ui.label('|').classes('text-gray-400')
                                progress_label = ui.label('Marked: 0 frames').classes('text-sm')
                                export_status = ui.label('').classes('text-xs text-purple-500 ml-2')

                            # Hidden file input for CSV loading
                            load_csv_upload = ui.upload(on_upload=lambda e: handle_csv_upload(e), auto_upload=True).props('accept=.csv').classes('hidden')

                        # Plot for results
                        video_plot = ui.card().classes('w-full mt-2 hidden')

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
        
        sim = VolleyballSimulation(h, D, v0, alpha, c, m=m)
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
    
    # --- VIDEO ANALYSIS LOGIC ---
    async def handle_upload(e: events.UploadEventArguments):
        # Store original filename (without extension)
        original_name = e.file.name if hasattr(e.file, 'name') else 'video'
        if '.' in original_name:
            original_name = original_name.rsplit('.', 1)[0]
        current_video_name['name'] = original_name

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            temp_path = f.name

        # Use async save method from NiceGUI 3.x FileUpload
        await e.file.save(temp_path)

        try:
            analyzer.load_video(temp_path)
            # Get first frame
            frame = analyzer.get_frame(0)
            if frame is not None:
                # Convert to base64 for interactive image
                _, im_arr = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                im_b64 = base64.b64encode(im_arr).decode('utf-8')
                interactive_image.source = f'data:image/jpeg;base64,{im_b64}'
                interactive_image.content = ''  # Clear any previous SVG

                status_label.text = f'Video Loaded ({analyzer.frame_count} frames, {analyzer.fps:.1f} fps). ZOOM in and CLICK edges of the ball.'
                calib_card.classes(remove='hidden')
                frame_editor_card.classes(remove='hidden') # Keep controls visible for zoom/nav
                ui.notify('Video loaded. Please zoom in and calibrate scale.')
                
                # Reset calibration state
                calib_state['points'] = []
                calib_state['active'] = True
                
        except Exception as err:
            ui.notify(f"Error loading video: {err}", type='negative')


    # Frame browser state
    playing_video = {'active': False}
    current_browse_frame = {'idx': 0}
    
    # Calibration State
    calib_state = {'active': False, 'points': [], 'origin_mode': False}

    # Zoom state for coordinate conversion
    zoom_state = {'factor': 1.0, 'offset_x': 0, 'offset_y': 0}

    @safe_handler
    def show_frame(frame_idx):
        """Display a specific frame with tracking overlay if available"""
        current_browse_frame['idx'] = frame_idx

        frame = analyzer.get_frame(frame_idx)
        if frame is None:
            return

        # Convert to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Check if we have tracking data for this frame
        results = tracking_data['results']
        has_tracking = len(results) > 0

        # Find tracking data for this frame number
        tracked_pos = None
        for r in results:
            if r['frame'] == frame_idx:
                tracked_pos = (r['x_pixel'], r['y_pixel'])
                break

        # Get current ball position for zoom centering
        # Priority: 1) marked position for this frame, 2) interpolated from nearby
        if tracked_pos:
            cx, cy = int(tracked_pos[0]), int(tracked_pos[1])
        elif has_tracking:
            # Interpolate position from nearby marked frames
            prev_pos = None
            next_pos = None
            for r in results:
                if r['frame'] < frame_idx:
                    prev_pos = r
                elif r['frame'] > frame_idx and next_pos is None:
                    next_pos = r
                    break

            if prev_pos and next_pos:
                # Linear interpolation between prev and next
                t = (frame_idx - prev_pos['frame']) / (next_pos['frame'] - prev_pos['frame'])
                cx = int(prev_pos['x_pixel'] + t * (next_pos['x_pixel'] - prev_pos['x_pixel']))
                cy = int(prev_pos['y_pixel'] + t * (next_pos['y_pixel'] - prev_pos['y_pixel']))
            elif prev_pos:
                # Extrapolate from last known position
                cx, cy = int(prev_pos['x_pixel']), int(prev_pos['y_pixel'])
            elif next_pos:
                cx, cy = int(next_pos['x_pixel']), int(next_pos['y_pixel'])
            else:
                cx, cy = None, None
        else:
            cx, cy = None, None

        # Draw trajectory (all marked points up to current frame)
        if has_tracking:
            # Find all points up to current frame
            points_to_draw = []
            for i, r in enumerate(results):
                if r['frame'] > frame_idx:
                    break
                points_to_draw.append(r)

            # Draw all points - small dots so they don't cover the ball
            for j, r in enumerate(points_to_draw):
                px = int(r['x_pixel'])
                py = int(r['y_pixel'])
                # Draw trail line to previous point
                if j > 0:
                    prev_r = points_to_draw[j-1]
                    prev_px = int(prev_r['x_pixel'])
                    prev_py = int(prev_r['y_pixel'])
                    cv2.line(frame_bgr, (prev_px, prev_py), (px, py), (0, 255, 0), 1)
                # Draw small point - tiny red for current frame, tiny green for others
                if r['frame'] == frame_idx:
                    cv2.drawMarker(frame_bgr, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
                else:
                    cv2.circle(frame_bgr, (px, py), 2, (0, 255, 0), -1)

        # Draw small crosshair at current position (not filled, just outline)
        if cx is not None and not tracked_pos:
            # Only draw if this frame isn't already marked (avoid covering the ball)
            cv2.drawMarker(frame_bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

        # Draw instruction text
        if calib_state['active']:
             msg = 'Click point 1' if len(calib_state['points']) == 0 else 'Click point 2'
             cv2.putText(frame_bgr, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             # Draw points so far - use small crosshairs for precision
             for p in calib_state['points']:
                 px, py = int(p[0]), int(p[1])
                 # Small crosshair (3 pixels each direction) instead of filled circle
                 cv2.drawMarker(frame_bgr, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)
                 # Tiny dot at exact center
                 cv2.circle(frame_bgr, (px, py), 1, (0, 255, 255), -1)
        elif calib_state['origin_mode']:
             cv2.putText(frame_bgr, 'Click Origin (0,0)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
             cv2.putText(frame_bgr, 'Click on ball to mark/correct', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Apply zoom (use center of frame if no ball position)
        zoom_factor = zoom_slider.value
        if zoom_factor > 1:
            h, w = frame_bgr.shape[:2]
            # Default to center of frame if no ball position
            if cx is None:
                cx, cy = w // 2, h // 2
            crop_w = max(10, int(w / zoom_factor))
            crop_h = max(10, int(h / zoom_factor))
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)

            crop_x1 = int(cx - crop_w // 2)
            crop_y1 = int(cy - crop_h // 2)
            crop_x1 = max(0, min(crop_x1, w - crop_w))
            crop_y1 = max(0, min(crop_y1, h - crop_h))
            crop_x2 = min(crop_x1 + crop_w, w)
            crop_y2 = min(crop_y1 + crop_h, h)

            zoom_state['factor'] = zoom_factor
            zoom_state['offset_x'] = crop_x1
            zoom_state['offset_y'] = crop_y1

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                cropped = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                if cropped.size > 0:
                    frame_bgr = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            zoom_state['factor'] = 1.0
            zoom_state['offset_x'] = 0
            zoom_state['offset_y'] = 0

        # Convert to base64 and display
        _, im_arr = cv2.imencode('.jpg', frame_bgr)
        im_b64 = base64.b64encode(im_arr).decode('utf-8')
        interactive_image.source = f'data:image/jpeg;base64,{im_b64}'
        interactive_image.content = ''  # Clear SVG overlay

        # Update labels
        time_s = frame_idx / analyzer.fps
        pos_str = f' | Ball: ({cx}, {cy})' if cx else ''
        tracked_str = ' [TRACKED]' if tracked_pos else ''
        zoom_str = f' | Zoom: {zoom_factor}x' if zoom_factor > 1 else ''
        frame_label.text = f'Frame: {frame_idx} / {analyzer.frame_count} | Time: {time_s:.2f}s{pos_str}{tracked_str}{zoom_str}'
        zoom_label.text = f'{zoom_factor}x'

    # Debounce for frame browser slider - only update on mouse release
    frame_browser_debounce = {'last_value': -1, 'pending': None}

    @safe_handler
    def on_frame_slider_change(e):
        """Called on slider drag - just update the label, don't load frame"""
        # Stop playback if running
        if playing_video['active']:
            playing_video['active'] = False
            play_btn.text = '▶ Play'

        idx = int(e.args) if isinstance(e.args, (int, float)) else int(frame_slider.value)
        frame_browser_debounce['pending'] = idx
        # Update label immediately so user sees where they're going
        frame_label.text = f'Frame: {idx} / {analyzer.frame_count} (release to load)'

    @safe_handler
    def on_frame_slider_release(_e):
        """Called when slider is released - now load the frame"""
        idx = frame_browser_debounce['pending']
        if idx is not None and idx != frame_browser_debounce['last_value']:
            frame_browser_debounce['last_value'] = idx
            frame_input.set_value(idx)
            show_frame(idx)

    # Use 'change' event which fires on release, not during drag
    frame_slider.on('update:model-value', on_frame_slider_change)
    frame_slider.on('change', on_frame_slider_release)

    @safe_handler
    def on_frame_input_change(_e):
        """Direct frame number input - jump immediately"""
        # Stop playback if running
        if playing_video['active']:
            playing_video['active'] = False
            play_btn.text = '▶ Play'

        idx = int(frame_input.value) if frame_input.value is not None else 0
        idx = max(0, min(idx, analyzer.frame_count - 1))
        if idx != frame_browser_debounce['last_value']:
            frame_browser_debounce['last_value'] = idx
            frame_slider.set_value(idx)
            show_frame(idx)

    frame_input.on('change', on_frame_input_change)

    @safe_handler
    def go_prev_frame():
        """Go to previous frame"""
        idx = current_browse_frame['idx']
        if idx > 0:
            new_idx = idx - 1
            frame_slider.set_value(new_idx)
            frame_input.set_value(new_idx)
            frame_browser_debounce['last_value'] = new_idx
            show_frame(new_idx)

    @safe_handler
    def go_next_frame():
        """Go to next frame"""
        idx = current_browse_frame['idx']
        if idx < analyzer.frame_count - 1:
            new_idx = idx + 1
            frame_slider.set_value(new_idx)
            frame_input.set_value(new_idx)
            frame_browser_debounce['last_value'] = new_idx
            show_frame(new_idx)

    prev_frame_btn.on_click(go_prev_frame)
    next_frame_btn.on_click(go_next_frame)

    # Zoom slider handler
    @safe_handler
    def on_zoom_change(_e):
        """Refresh display when zoom changes"""
        show_frame(current_browse_frame['idx'])

    zoom_slider.on('update:model-value', on_zoom_change)

    @safe_async_handler
    async def play_video_frames():
        """Play through video frames"""
        playing_video['active'] = True
        try:
            play_btn.text = '⏸ Pause'

            start_idx = current_browse_frame['idx']
            for i in range(start_idx, analyzer.frame_count):
                if not playing_video['active']:
                    break
                frame_slider.set_value(i)
                show_frame(i)
                await asyncio.sleep(1.0 / analyzer.fps)  # Play at video fps

            playing_video['active'] = False
            play_btn.text = '▶ Play'
        except RuntimeError:
            # Client disconnected
            playing_video['active'] = False

    def toggle_play():
        if playing_video['active']:
            playing_video['active'] = False
            play_btn.text = '▶ Play'
        else:
            asyncio.create_task(play_video_frames())

    play_btn.on_click(toggle_play)

    # Store manual ball positions - simple list sorted by frame
    tracking_data = {'results': []}

    def update_progress_label():
        """Update the progress label with current marking count"""
        count = len(tracking_data['results'])
        progress_label.text = f'Marked: {count} frames'
        
    def set_scale():
        if len(calib_state['points']) == 2:
           try:
               p1 = calib_state['points'][0]
               p2 = calib_state['points'][1]
               real_dist = real_dist_input.value
               analyzer.set_calibration(p1, p2, real_dist)

               ui.notify(f"Scale set: {analyzer.pixels_per_meter:.2f} px/m. Now click to set origin.", type='positive')
               calib_state['points'] = []
               calib_state['active'] = False
               calib_btn.disable()
               # Automatically enter origin mode
               calib_state['origin_mode'] = True
               show_frame(current_browse_frame['idx'])
           except ValueError as e:
                ui.notify(str(e), type='warning')

    def enable_origin_mode():
        calib_state['origin_mode'] = True
        show_frame(current_browse_frame['idx'])
        ui.notify('Click a single point to set origin')
        
    @safe_handler
    def on_image_click(e: events.MouseEventArguments):
        frame_idx = current_browse_frame['idx']

        # Convert click coordinates from zoomed view back to original
        click_x = e.image_x
        click_y = e.image_y

        if zoom_state['factor'] > 1:
            frame = analyzer.get_frame(frame_idx)
            if frame is not None:
                h, w = frame.shape[:2]
                click_x = click_x / zoom_state['factor'] + zoom_state['offset_x']
                click_y = click_y / zoom_state['factor'] + zoom_state['offset_y']
                
        # --- CALIBRATION MODE ---
        if calib_state['active']:
            calib_state['points'].append((click_x, click_y))
            if len(calib_state['points']) > 2:
                 calib_state['points'] = calib_state['points'][-2:]
            
            show_frame(frame_idx) # update to draw points
            
            if len(calib_state['points']) == 2:
                calib_btn.enable()
            else:
                calib_btn.disable()
            return
            
        elif calib_state['origin_mode']:
            analyzer.set_origin((click_x, click_y))
            calib_state['origin_mode'] = False
            ui.notify(f'Origin set to ({click_x:.0f}, {click_y:.0f})')
            status_label.text = f'Scale: {analyzer.pixels_per_meter:.2f} px/m, Origin: ({click_x:.0f}, {click_y:.0f}). Ready to track.'
            calib_card.classes(add='hidden')
            # frame_editor_card is already visible
            show_frame(frame_idx)
            return

        # Check if we already have data for this frame
        results = tracking_data['results']
        existing_idx = None
        for i, r in enumerate(results):
            if r['frame'] == frame_idx:
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing position
            results[existing_idx]['x_pixel'] = click_x
            results[existing_idx]['y_pixel'] = click_y
            correction_status.text = f'Frame {frame_idx}: Updated to ({click_x:.0f}, {click_y:.0f})'
        else:
            # Add new position
            new_entry = {
                'frame': frame_idx,
                'time': frame_idx / analyzer.fps,
                'x_pixel': click_x,
                'y_pixel': click_y
            }
            # Insert in sorted order by frame
            inserted = False
            for i, r in enumerate(results):
                if r['frame'] > frame_idx:
                    results.insert(i, new_entry)
                    inserted = True
                    break
            if not inserted:
                results.append(new_entry)

            correction_status.text = f'Frame {frame_idx}: Marked at ({click_x:.0f}, {click_y:.0f})'

        # Update progress
        update_progress_label()

        # Refresh current frame display
        show_frame(frame_idx)

        # Auto-advance to next frame
        if frame_idx < analyzer.frame_count - 1:
            new_idx = frame_idx + 1
            frame_slider.set_value(new_idx)
            show_frame(new_idx)

    interactive_image.on_mouse(on_image_click)

    @safe_handler
    def clear_all_marks():
        """Clear all marked positions"""
        tracking_data['results'] = []
        update_progress_label()
        show_frame(current_browse_frame['idx'])
        ui.notify('All marks cleared')

    clear_btn.on_click(clear_all_marks)

    @safe_handler
    def finish_and_export():
        """Finalize tracking and export data to CSV"""
        try:
            results = tracking_data['results']
            if not results:
                ui.notify('No tracking data to export', type='warning')
                return

            # Sort by frame number to handle out-of-order marking
            results_sorted = sorted(results, key=lambda r: r['frame'])

            # Create CSV content
            # If calibrated, we can also compute real-world meters relative to origin
            has_calib = analyzer.is_calibrated
            header = 'frame,time_s,x_pixel,y_pixel'
            if has_calib:
                header += ',x_meter,y_meter'
            
            csv_lines = [header]

            for r in results_sorted:
                line = f"{r['frame']},{r['time']:.4f},{r['x_pixel']:.1f},{r['y_pixel']:.1f}"
                if has_calib:
                    # Transform pixel to meter
                    # x_m = (x_pix - origin_x) / px_per_m
                    # y_m = (origin_y - y_pix) / px_per_m  (since y pixel grows downwards, but meters grow upwards)
                    xm = (r['x_pixel'] - analyzer.origin[0]) / analyzer.pixels_per_meter
                    ym = (analyzer.origin[1] - r['y_pixel']) / analyzer.pixels_per_meter
                    line += f",{xm:.4f},{ym:.4f}"
                csv_lines.append(line)

            csv_content = '\n'.join(csv_lines)

            # Save to current working directory with video name
            video_name = current_video_name['name'] or 'tracking_data'
            export_filename = f'{video_name}_tracking.csv'
            export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_filename)
            with open(export_path, 'w') as f:
                f.write(csv_content)

            # Also print summary to console
            print("\n" + "="*60)
            print("TRACKING DATA EXPORTED")
            print("="*60)
            print(f"File: {export_path}")
            print(f"Total frames marked: {len(results_sorted)}")
            print(f"Time range: {results_sorted[0]['time']:.2f}s to {results_sorted[-1]['time']:.2f}s")

            # Calculate trajectory stats (using sorted data)
            x_pixels = [r['x_pixel'] for r in results_sorted]
            y_pixels = [r['y_pixel'] for r in results_sorted]
            print(f"X range: {min(x_pixels):.0f} to {max(x_pixels):.0f} pixels")
            print(f"Y range: {min(y_pixels):.0f} to {max(y_pixels):.0f} pixels")
            print("="*60 + "\n")

            # Show trajectory shape graph
            video_plot.classes(remove='hidden')

            # Calculate ranges to detect motion type
            x_min, x_max = min(x_pixels), max(x_pixels)
            y_min, y_max = min(y_pixels), max(y_pixels)
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1

            # Detect motion type: vertical if Y range >> X range
            is_vertical_motion = y_range > x_range * 3

            with video_plot:
                video_plot.clear()
                with ui.pyplot(figsize=(10, 6)) as plot:
                    fig = plot.fig
                    ax = fig.add_subplot(1, 1, 1)

                    if is_vertical_motion:
                        # Vertical motion: plot Time vs Height
                        times = [r['time'] for r in results_sorted]
                        # Invert Y so up is up (lower pixel = higher position)
                        heights = [(y_max - yp) for yp in y_pixels]

                        ax.scatter(times, heights, c='green', s=30, zorder=4, label='Marked points')
                        ax.plot(times[0], heights[0], 'go', markersize=12, label='Start', zorder=5)
                        ax.plot(times[-1], heights[-1], 'ro', markersize=12, label='End', zorder=5)

                        ax.set_xlabel('Time (seconds)')
                        ax.set_ylabel('Height (pixels from bottom)')
                        ax.set_title('Ball Height vs Time (Vertical Motion)')
                    else:
                        # Horizontal/diagonal motion: plot X vs Y
                        x_norm = [(xp - x_min) / x_range for xp in x_pixels]
                        y_norm = [(y_max - yp) / y_range for yp in y_pixels]

                        ax.scatter(x_norm, y_norm, c='green', s=30, zorder=4, label='Marked points')
                        ax.plot(x_norm[0], y_norm[0], 'go', markersize=12, label='Start', zorder=5)
                        ax.plot(x_norm[-1], y_norm[-1], 'ro', markersize=12, label='End', zorder=5)

                        ax.set_xlabel('Horizontal Position (normalized)')
                        ax.set_ylabel('Height (normalized)')
                        ax.set_title('Ball Trajectory Shape')
                        ax.set_xlim(-0.05, 1.05)
                        ax.set_ylim(-0.05, 1.05)

                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    fig.tight_layout()

            export_status.text = f'Saved to {export_path} ({len(results_sorted)} points)'
            ui.notify(f'Tracking data exported to {export_path}', type='positive')

        except Exception as err:
            print(f"Export error: {err}")
            traceback.print_exc()
            ui.notify(f'Export failed: {err}', type='negative')

    finish_btn.on_click(finish_and_export)

    # Wire up Load CSV button to trigger the hidden file upload
    load_csv_btn.on_click(lambda: load_csv_upload.run_method('pickFiles'))

    @safe_async_handler
    async def handle_csv_upload(e: events.UploadEventArguments):
        """Load tracking data from a CSV file"""
        try:
            # Read CSV content
            content = await e.file.read()
            csv_text = content.decode('utf-8')
            lines = csv_text.strip().split('\n')

            if len(lines) < 2:
                ui.notify('CSV file is empty or has no data', type='warning')
                return

            # Check header
            header = lines[0].strip()
            if header != 'frame,time_s,x_pixel,y_pixel':
                ui.notify(f'Invalid CSV format. Expected header: frame,time_s,x_pixel,y_pixel', type='warning')
                return

            # Parse data rows
            new_results = []
            for i, line in enumerate(lines[1:], start=2):
                parts = line.strip().split(',')
                if len(parts) != 4:
                    ui.notify(f'Invalid row {i}: {line}', type='warning')
                    continue
                try:
                    frame = int(parts[0])
                    time_s = float(parts[1])
                    x_pixel = float(parts[2])
                    y_pixel = float(parts[3])
                    new_results.append({
                        'frame': frame,
                        'time': time_s,
                        'x_pixel': x_pixel,
                        'y_pixel': y_pixel
                    })
                except ValueError as ve:
                    ui.notify(f'Error parsing row {i}: {ve}', type='warning')
                    continue

            if not new_results:
                ui.notify('No valid data found in CSV', type='warning')
                return

            # Sort by frame and load into tracking_data
            new_results.sort(key=lambda r: r['frame'])
            tracking_data['results'] = new_results

            # Update progress label
            update_progress_label()

            # Show trajectory graph
            video_plot.classes(remove='hidden')
            x_pixels = [r['x_pixel'] for r in new_results]
            y_pixels = [r['y_pixel'] for r in new_results]

            # Calculate ranges to detect motion type
            x_min, x_max = min(x_pixels), max(x_pixels)
            y_min, y_max = min(y_pixels), max(y_pixels)
            x_range = x_max - x_min if x_max != x_min else 1
            y_range = y_max - y_min if y_max != y_min else 1

            # Detect motion type: vertical if Y range >> X range
            is_vertical_motion = y_range > x_range * 3

            with video_plot:
                video_plot.clear()
                with ui.pyplot(figsize=(10, 6)) as plot:
                    fig = plot.fig
                    ax = fig.add_subplot(1, 1, 1)

                    if is_vertical_motion:
                        # Vertical motion: plot Time vs Height
                        times = [r['time'] for r in new_results]
                        # Invert Y so up is up (lower pixel = higher position)
                        heights = [(y_max - yp) for yp in y_pixels]

                        ax.scatter(times, heights, c='green', s=30, zorder=4, label='Marked points')
                        ax.plot(times[0], heights[0], 'go', markersize=12, label='Start', zorder=5)
                        ax.plot(times[-1], heights[-1], 'ro', markersize=12, label='End', zorder=5)

                        ax.set_xlabel('Time (seconds)')
                        ax.set_ylabel('Height (pixels from bottom)')
                        ax.set_title('Ball Height vs Time (Vertical Motion - Loaded from CSV)')
                    else:
                        # Horizontal/diagonal motion: plot X vs Y
                        x_norm = [(xp - x_min) / x_range for xp in x_pixels]
                        y_norm = [(y_max - yp) / y_range for yp in y_pixels]

                        ax.scatter(x_norm, y_norm, c='green', s=30, zorder=4, label='Marked points')
                        ax.plot(x_norm[0], y_norm[0], 'go', markersize=12, label='Start', zorder=5)
                        ax.plot(x_norm[-1], y_norm[-1], 'ro', markersize=12, label='End', zorder=5)

                        ax.set_xlabel('Horizontal Position (normalized)')
                        ax.set_ylabel('Height (normalized)')
                        ax.set_title('Ball Trajectory Shape (Loaded from CSV)')
                        ax.set_xlim(-0.05, 1.05)
                        ax.set_ylim(-0.05, 1.05)

                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    fig.tight_layout()

        except Exception as err:
            ui.notify(f"Error handling CSV upload: {err}", type='negative')
            print(f"Error handling CSV upload: {err}")
            traceback.print_exc()

    ui.run(title=ENGLISH_TEXT['title'])

if __name__ in {"__main__", "__mp_main__"}:
    main()