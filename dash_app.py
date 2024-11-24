# app.py

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import json

# Load precomputed data
with open('rainbow_data_3D.json', 'r') as file:
    data = json.load(file)

# Create the Dash app
app = Dash(__name__)

# Define the sphere (water droplet) for visualization
radius = 1.0  # Should match the radius used in calculate.py
sphere_center = np.array([0.0, 0.0, 0.0])  # Ensure consistency

# Create the sphere surface for visualization
theta_sphere = np.linspace(0, 2 * np.pi, 50)
phi_sphere = np.linspace(0, np.pi, 50)
theta_sphere, phi_sphere = np.meshgrid(theta_sphere, phi_sphere)

x_sphere = sphere_center[0] + radius * np.sin(phi_sphere) * np.cos(theta_sphere)
y_sphere = sphere_center[1] + radius * np.sin(phi_sphere) * np.sin(theta_sphere)
z_sphere = sphere_center[2] + radius * np.cos(phi_sphere)

sphere_trace = go.Surface(
    x=x_sphere,
    y=y_sphere,
    z=z_sphere,
    colorscale='Blues',
    opacity=0.5,
    showscale=False,
    name='Water Droplet',
    hoverinfo='skip'
)

# Extract available Sun positions from the precomputed data
azimuth_angles = sorted({int(key.split('_')[1]) for key in data.keys()})
elevation_angles = sorted({int(key.split('_')[3]) for key in data.keys()})

# Since the cylinder radius is fixed in the precomputed data, we cannot adjust it dynamically

# Layout of the app with sliders for Sun's position
app.layout = html.Div([
    html.H1('3D Rainbow Visualization with Precomputed Data'),
    html.Label('Sun Azimuth Angle θ (degrees)'),
    dcc.Slider(
        id='theta-slider',
        min=min(azimuth_angles),
        max=max(azimuth_angles),
        step=30,
        value=min(azimuth_angles),
        marks={i: str(i) for i in azimuth_angles},
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='mouseup'  # Update on mouse release to improve performance
    ),
    html.Label('Sun Elevation Angle φ (degrees)'),
    dcc.Slider(
        id='phi-slider',
        min=min(elevation_angles),
        max=max(elevation_angles),
        step=30,
        value=min(elevation_angles),
        marks={i: str(i) for i in elevation_angles},
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='mouseup'  # Update on mouse release to improve performance
    ),
    dcc.Graph(id='rainbow-plot'),
])

# Callback to update the plot based on the Sun's position
@app.callback(
    Output('rainbow-plot', 'figure'),
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value')
)
def update_plot(theta_value, phi_value):
    # Construct the key to access data
    key = f"theta_{theta_value}_phi_{phi_value}"

    # Prepare traces
    traces = [sphere_trace]

    # Compute the incoming light direction vector D
    theta_rad = np.radians(theta_value)
    phi_rad = np.radians(phi_value)
    D = np.array([
        np.cos(phi_rad) * np.cos(theta_rad),
        np.cos(phi_rad) * np.sin(theta_rad),
        np.sin(phi_rad)
    ])
    D /= np.linalg.norm(D)

    # Add Sun representation in the plot
    sun_distance = 10  # Distance from the sphere to the Sun
    sun_position = sphere_center - D * sun_distance
    sun_trace = go.Scatter3d(
        x=[sun_position[0]],
        y=[sun_position[1]],
        z=[sun_position[2]],
        mode='markers',
        marker=dict(size=5, color='yellow'),
        name='Sun',
        hoverinfo='skip'
    )
    traces.append(sun_trace)

    # Check if the key exists in data
    if key in data:
        data_entry = data[key]
        rays = data_entry["rays"]

        # Add rays to the plot
        for ray in rays:
            color = ray['color']
            path = ray['path']
            for segment in ['incoming', 'inside1', 'inside2', 'outgoing']:
                if segment in path:
                    points = path[segment]
                    if len(points) >= 2:
                        x_vals = [point[0] for point in points]
                        y_vals = [point[1] for point in points]
                        z_vals = [point[2] for point in points]
                        trace = go.Scatter3d(
                            x=x_vals,
                            y=y_vals,
                            z=z_vals,
                            mode='lines',
                            line=dict(color=color, width=2),
                            hoverinfo='skip',
                            showlegend=False
                        )
                        traces.append(trace)
    else:
        # Handle missing data
        print(f"No data available for θ={theta_value}°, φ={phi_value}°")
        # Optionally, you could display a message or plot nothing

    # Create the figure with all traces
    fig = go.Figure(data=traces)

    # Update layout settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True, range=[-10, 10], title='X'),
            yaxis=dict(visible=True, range=[-10, 10], title='Y'),
            zaxis=dict(visible=True, range=[-10, 10], title='Z'),
            aspectmode='cube'  # Ensures the aspect ratio is equal on all axes
        ),
        title=f'Sun Position θ={theta_value}°, φ={phi_value}°',
        showlegend=False
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
