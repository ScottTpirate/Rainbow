import json
import numpy as np
import plotly.graph_objects as go

# Load ray data from JSON file
with open('ray_data.json', 'r') as json_file:
    ray_data = json.load(json_file)

# Create sphere surface
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x_sphere = np.cos(u)*np.sin(v)
y_sphere = np.sin(u)*np.sin(v)
z_sphere = np.cos(v)

fig = go.Figure()

# Add sphere surface
sphere_trace = go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    opacity=0.5,
    colorscale='Blues',
    showscale=False,
    name='Water Droplet'
)

# Define colors for different wavelengths
color_map = {
    'red': 'red',
    'green': 'green',
    'blue': 'blue'
}

frames = []

for level_str, level_data in ray_data.items():
    data = [sphere_trace]
    for color, path in level_data.items():
        points = np.array(path['points'])
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        exit_dir = np.array(path['directions'][3])
        exit_point = points[3]
        t_exit = 5  # Arbitrary value to extend the exiting ray
        exit_ray_end = exit_point + exit_dir * t_exit

        # Collect traces for this level and color
        traces = [
            go.Scatter3d(
                x=[points[0][0], points[1][0]],
                y=[points[0][1], points[1][1]],
                z=[points[0][2], points[1][2]],
                mode='lines',
                line=dict(color=color_map[color], width=3),
                name=f'Incoming {color} ray at level {float(level_str):.2f}'
            ),
            go.Scatter3d(
                x=[points[1][0], points[2][0]],
                y=[points[1][1], points[2][1]],
                z=[points[1][2], points[2][2]],
                mode='lines',
                line=dict(color=color_map[color], width=3, dash='dash'),
                showlegend=False
            ),
            go.Scatter3d(
                x=[points[2][0], points[3][0]],
                y=[points[2][1], points[3][1]],
                z=[points[2][2], points[3][2]],
                mode='lines',
                line=dict(color=color_map[color], width=3, dash='dot'),
                showlegend=False
            ),
            go.Scatter3d(
                x=[exit_point[0], exit_ray_end[0]],
                y=[exit_point[1], exit_ray_end[1]],
                z=[exit_point[2], exit_ray_end[2]],
                mode='lines',
                line=dict(color=color_map[color], width=3),
                showlegend=False
            )
        ]
        data.extend(traces)
    frame = go.Frame(data=data, name=level_str)
    frames.append(frame)

# Add the first frame's data
fig.add_traces(frames[0].data)

# Update figure with frames
fig.update(frames=frames)

# Create slider steps
slider_steps = []
for i, frame in enumerate(frames):
    slider_step = {
        'args': [
            [frame.name],
            {'frame': {'duration': 0},  
            'mode': 'immediate',
            'transition': {'duration': 0}}
        ],
        'label': f'{float(frame.name):.2f}',
        'method': 'animate'
    }
    slider_steps.append(slider_step)

# Configure sliders
sliders = [{
    'active': 0,
    'currentvalue': {'prefix': 'Entry Level: '},
    'pad': {'t': 50},
    'steps': slider_steps
}]

x_range = [-2, 2]
y_range = [-2, 2]
z_range = [-2, 2]

# Update layout with sliders
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X', range=x_range),
        yaxis=dict(title='Y', range=y_range),
        zaxis=dict(title='Z', range=z_range),
        aspectratio=dict(x=1, y=1, z=1)
    ),
    sliders=sliders,
    updatemenus=[{
        'type': 'buttons',
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 500},  # Set redraw to False
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig.show()
