import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import subprocess
import os

# Represents a dynamic visualization of gaze data for a specific stimulus video
class VideoGazeVisualization:
    def __init__(self, video_path, canvas_size=(1680, 1050), bg_color=(211, 211, 211)):
        """
        video_path: path to stimulus video that visualizations will be overlaid over
        canvas_size: size of canvas that video will be centered in
        bg_color: color of background that video doesn't cover, defaults to gray
        """
        self.video_path = video_path
        self.canvas_w, self.canvas_h = canvas_size
        self.bg_color = bg_color
        self.legend = {}

        # Read video metadata without loading the whole file into memory
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        x_offset = (self.canvas_w - vid_w) // 2
        y_offset = (self.canvas_h - vid_h) // 2

        self.cy1 = max(0, y_offset)
        self.cy2 = min(self.canvas_h, y_offset + vid_h)
        self.cx1 = max(0, x_offset)
        self.cx2 = min(self.canvas_w, x_offset + vid_w)

        self.vy1 = max(0, -y_offset)
        self.vy2 = self.vy1 + (self.cy2 - self.cy1)
        self.vx1 = max(0, -x_offset)
        self.vx2 = self.vx1 + (self.cx2 - self.cx1)

        # A list to store our requested visualizations (Deferred Execution)
        self.layers = []

    def add_heatmap(self, label, data, sigma=20, time_window_ms=500, alpha=0.8, mods='', colormap=cv2.COLORMAP_INFERNO):
        """
        Queues a heatmap visualization.
        data: list of lists of tuples (t, x, y, w (optional)) representing the data to generate the heatmap from
        time_window_ms: If set to 1000, shows the heatmap of data 1 second prior to the current frame.
                        If None, shows a cumulative heatmap of all data up to the current frame.
        """

        flat = np.vstack(data)
        timestamps = flat[:, 0]
        self.layers.append({
            'type': 'heatmap',
            'label': label,
            'sigma': sigma,
            'window': time_window_ms,
            'alpha': alpha,
            'modifications': mods,
            'colormap': colormap,
            'flat_data': flat,
            'timestamps': timestamps,
        })
        return self

    def add_trace(self, label, data, fade_factor=0.85, color=(255, 255, 0), thickness=2, max_gap_ms=150):
        """
        Adds a fading comet tail trace for the individuals in the given data
        label: group label
        data: list of lists of tuples (t, x, y) representing the data to add traces for
        fade_factor: Multiplier applied every frame, lower = faster fade
        color: BGR typle. Default is cyan.
        max_gap_ms: Prevents drawing giant lines if a user blinks or tracker loses them
        """
        data = [np.array(user) for user in data]
        self.layers.append({
            'type': 'trace',
            'label': label,
            'fade_factor': fade_factor,
            'color': color,
            'thickness': thickness,
            'max_gap_ms': max_gap_ms,
            'indiv_data': data,
        })

        self.legend[label] = color
        return self

    def save(self, output_path, is_test=False):
        """
        Saves final video as an mp4 with all added layers to output_path (path should contain /[output_name].mp4).
        is_test: If True, only processes the first 100 frames of the video
        """
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.canvas_w, self.canvas_h))

        # --- SETUP LAYER STATES ---
        trace_accumulators = {}
        last_points = {}
        for idx, layer in enumerate(self.layers):
            if layer['type'] == 'trace':
                # 1-channel array (grayscale) holding only Opacity
                trace_accumulators[idx] = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
                # last_points[idx][i] = last data point for user i that was rendered in form (t, x, y)
                last_points[idx] = [None] * len(layer['indiv_data'])

        frame_count = 0
        prev_time_ms = -1.0 

        # loop until 100th frame for testing or until end of stimulus video
        while (is_test and frame_count < 100) or (not is_test and True):
            ret, frame = cap.read()
            if not ret: break
            
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # 2d array representation of current pixels
            canvas = np.full((self.canvas_h, self.canvas_w, 3), self.bg_color, dtype=np.uint8)
            # Set center of canvas to contain pixels of current frame
            canvas[self.cy1:self.cy2, self.cx1:self.cx2] = frame[self.vy1:self.vy2, self.vx1:self.vx2]

            # Loop through visualization layers
            for idx, layer in enumerate(self.layers):
                
                # --- TRACE LAYER ---
                if layer['type'] == 'trace':
                    # 2d array storing opacities of traces at each pixel of frame
                    accum = trace_accumulators[idx]
                    
                    # Dim the opacity of old traces
                    accum *= layer['fade_factor']
                    
                    # Create a temporary 2D mask for all individual's trace lines for current frame
                    current_lines = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)
                    
                    # Loop through individuals
                    for user_idx, user_data in enumerate(layer['indiv_data']):
                        # Mask for data in current frame's ms time slot
                        mask = (user_data[:, 0] > prev_time_ms) & (user_data[:, 0] <= current_time_ms)
                        new_points = user_data[mask]

                        # last point from current user that was rendered
                        pt_last = last_points[idx][user_idx]
                        
                        if len(new_points) > 0:
                            pts_to_draw = []
                            if pt_last is not None:
                                # If time between last and current point is within allowed duration, connects previous
                                # point to current trace
                                if new_points[0, 0] - pt_last[0] <= layer['max_gap_ms']:
                                    pts_to_draw.append((pt_last[1], pt_last[2]))
                            
                            for p in new_points:
                                pts_to_draw.append((p[1], p[2]))
                                
                            # Draw new segments at 255 (maximum opacity) on the 2D mask
                            for i in range(1, len(pts_to_draw)):
                                p1 = (int(pts_to_draw[i-1][0]), int(pts_to_draw[i-1][1]))
                                p2 = (int(pts_to_draw[i][0]), int(pts_to_draw[i][1]))
                                # Adds line to current_lines
                                cv2.line(current_lines, p1, p2, 255, layer['thickness'], cv2.LINE_AA)
                                
                            last_points[idx][user_idx] = (new_points[-1, 0], new_points[-1, 1], new_points[-1, 2])
                    
                    # Add new lines to the accumulator, capping it safely at 255
                    accum += current_lines.astype(np.float32)
                    np.clip(accum, 0, 255, out=accum)

                    # Convert our 0-255 opacity map into a 0.0-1.0 percentage mask
                    # The np.newaxis makes it a 3D array (H, W, 1) so it can broadcast against the video
                    alpha_mask = (accum / 255.0)[..., np.newaxis]
                    color_arr = np.array(layer['color'], dtype=np.float32)
                    
                    # Blend the trace color smoothly over the canvas
                    canvas = (canvas * (1 - alpha_mask) + color_arr * alpha_mask).astype(np.uint8)
                    canvas = self.draw_legend(canvas)

                # --- HEATMAP LAYER ---
                elif layer['type'] == 'heatmap':
                    start_t = 0 if layer['window'] is None else max(0, current_time_ms - layer['window'])
                    mask = (layer['timestamps'] >= start_t) & (layer['timestamps'] <= current_time_ms)
                    valid_data = layer['flat_data'][mask]

                    heatmap_array = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
                    for idx, tup in enumerate(valid_data):
                        x = tup[1]
                        y = tup[2]
                        if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                            if len(tup) == 3:
                                heatmap_array[int(y), int(x)] += 1
                            else:
                                heatmap_array[int(y), int(x)] += tup[3]

                    if np.max(heatmap_array) > 0:
                        smoothed = gaussian_filter(heatmap_array, layer['sigma'])
                        if(layer['modifications'] == 'sqrt'):
                            smoothed = np.sqrt(smoothed, out=smoothed)
                            
                        smoothed = np.uint8(255 * smoothed / np.max(smoothed))
                        heatmap_color = cv2.applyColorMap(smoothed, layer['colormap'])
                        intensity_mask = (smoothed / 255.0)[..., np.newaxis] 
                        canvas = (canvas * (1 - intensity_mask * layer['alpha']) + 
                                 heatmap_color * (intensity_mask * layer['alpha'])).astype(np.uint8)

            out.write(canvas)
            prev_time_ms = current_time_ms
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{self.total_frames} frames...")

        cap.release()
        out.release()
        
        # add original sound to output video
        temp_output_file = "temp_output.mp4"
        command = [
            "ffmpeg",
            "-y",  
            "-i", output_path,
            "-i", self.video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            temp_output_file
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(temp_output_file, output_path)

        except subprocess.CalledProcessError as e:
            print("An error occurred while running ffmpeg:")
            print(e.stderr)
            
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
                
        except FileNotFoundError:
            print("Error: ffmpeg is not installed or not added to your system's PATH.")

        return self

    def draw_legend(self, frame, start_x = 20, start_y = 30):
        """
        Draws a legend on a cv2 frame.
        
        frame: The cv2 image frame
        legend_items: Dictionary of { "Label": (B, G, R) }
        start_x: Top-left X coordinate for the legend
        start_y: Top-left Y coordinate for the legend
        """
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        line_height = 35
        
        box_width = 150
        box_height = len(self.legend) * line_height
        
        top_left = (start_x - 10, start_y - 25)
        bottom_right = (start_x + box_width, start_y - 25 + box_height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)

        current_y = start_y
        for label, color in self.legend.items():
            cv2.circle(frame, (start_x + 15, current_y - 5), 8, color, -1)
            
            cv2.putText(frame, label, (start_x + 40, current_y), 
                        font, font_scale, (255, 255, 255), font_thickness)
            
            current_y += line_height
            
        return frame