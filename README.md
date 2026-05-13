# scitl_gaze_visualization
A tool to visualize eye tracking data of individuals or groups over a stimulus video via heatmaps and traces.

Install required dependencies from environment.yml, e.g.:
```conda env create -file environment.yml```

The DynamicVisualization class expects gaze data as lists of tuples of form (t, x, y) where t is the time in ms, and x and y are the screen coordinates of the data point.
