# app.py

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import json
import os
from functools import lru_cache

# Create the Dash app
app = Dash(__name__)

# Define the sphere (water droplet) for visualization
radius = 1.0  # Sphere radius
sphere_center = np.array([0.0, 0.0, 0.0])

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

# Define available Sun positions and cylinder radii
azimuth_angles = np.arange(0, 361, 30)    # 0° to 360° in 30° increments
elevation_angles = np.arange(-90, 91, 30)  # -90° to 90° in 30° increments
cylinder_radii = np.arange(0.1, 1.1, 0.1)  # 0.1 to 2.0 in steps of 0.1

# Define Dropdown options
azimuth_options = [{'label': f"{int(angle)}°", 'value': int(angle)} for angle in azimuth_angles]
elevation_options = [{'label': f"{int(angle)}°", 'value': int(angle)} for angle in elevation_angles]

# Layout of the app with sliders for Sun's position and cylinder radius
app.layout = html.Div([
    html.H1('3D Rainbow Visualization with Precomputed Data', style={'textAlign': 'center'}),

    html.Div([
        # Azimuth Dropdown
        html.Div([
            html.Label('Sun Azimuth Angle θ (degrees)'),
            dcc.Dropdown(
                id='theta-dropdown',
                options=azimuth_options,
                value=120,
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'margin-right': '20px'}),

        # Elevation Dropdown
        html.Div([
            html.Label('Sun Elevation Angle φ (degrees)'),
            dcc.Dropdown(
                id='phi-dropdown',
                options=elevation_options,
                value=-30,
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'margin-right': '20px'}),
        
        # Cylinder Radius Slider
        html.Div([
            html.Label('Cylinder Radius'),
            dcc.Slider(
                id='cylinder-radius-slider',
                min=min(cylinder_radii),
                max=max(cylinder_radii),
                step=0.1,
                value=0.5,
                marks={int(i): str(round(i,1)) for i in cylinder_radii},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
        ], style={'width': '300px'}),
    ], style={
        'display': 'flex',
        'flex-direction': 'row',
        'justify-content': 'center',
        'align-items': 'center',
        'margin-bottom': '40px'
    }),


    # Graph Component
    dcc.Graph(
        id='rainbow-plot',
        style={
            'height': '800px',
            'width': '1000px',
            'margin': '0 auto'
        }
    ),
], style={'width': '90%', 'margin': '0 auto'})  # Center the entire layout

# Cache loaded JSON data to avoid reloading from disk
@lru_cache(maxsize=100)
def load_data(theta_value):
    filename = f'rainbow_data_theta_{theta_value}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    else:
        print(f"Data file {filename} not found.")
        return None

 # Synchronous Callback to update the plot
@app.callback(
    Output('rainbow-plot', 'figure'),
    [Input('theta-dropdown', 'value'),
     Input('phi-dropdown', 'value'),
     Input('cylinder-radius-slider', 'value')]
)
def update_plot(theta_value, phi_value, cylinder_radius):
    # Prepare traces
    traces = [sphere_trace]
    
    # Ensure correct types
    try:
        theta_value = int(theta_value)
        phi_value = int(phi_value)
        cylinder_radius = round(float(cylinder_radius), 1)
    except ValueError as e:
        print(f"Type conversion error: {e}")
        # Return an empty figure
        empty_fig = go.Figure(data=traces)
        return empty_fig
    
    # Load data for the selected azimuth angle
    data = load_data(theta_value)
    if data is None:
        print(f"No data available for θ={theta_value}°")
        empty_fig = go.Figure(data=traces)
        return empty_fig
    
    # Get the key for the current elevation angle
    phi_key = f"phi_{phi_value}"
    if phi_key not in data:
        print(f"No data available for φ={phi_value}°")
        empty_fig = go.Figure(data=traces)
        return empty_fig
    
    data_entry = data[phi_key]
    rays_data = data_entry.get("rays", {})
    
    # Get the key for the current cylinder radius
    cylinder_radius_key = f"radius_{cylinder_radius:.1f}"
    if cylinder_radius_key not in rays_data:
        print(f"No data available for Cylinder Radius {cylinder_radius}")
        empty_fig = go.Figure(data=traces)
        return empty_fig
    
    rays_list = rays_data[cylinder_radius_key]
    
    # Process the rays to create traces
    for ray in rays_list:
        color = ray.get('color', 'black')  # Default to black if color not specified
        path = ray.get('path', {})
        for segment in path.values():
            if not segment:
                continue  # Skip empty segments
            x_vals = [point[0] for point in segment]
            y_vals = [point[1] for point in segment]
            z_vals = [point[2] for point in segment]

            trace = go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines',
                line=dict(color=color, width=3),
                # opacity=0.7,  
                hoverinfo='skip',
                showlegend=False
            )
            traces.append(trace)
    
    # Add Sun representation in the plot
    sun_distance = 10  # Distance from the sphere to the Sun
    theta_rad = np.radians(theta_value)
    phi_rad = np.radians(phi_value)
    D = np.array([
        np.cos(phi_rad) * np.cos(theta_rad),
        np.cos(phi_rad) * np.sin(theta_rad),
        np.sin(phi_rad)
    ])
    D /= np.linalg.norm(D)
    sun_position = sphere_center - D * sun_distance
    sun_trace = go.Scatter3d(
        x=[sun_position[0]],
        y=[sun_position[1]],
        z=[sun_position[2]],
        mode='markers',
        marker=dict(size=10, color='yellow'),
        name='Sun',
        hoverinfo='skip'
    )
    traces.append(sun_trace)
    
    # Create the figure with all traces
    fig = go.Figure(data=traces)
    
    # Update layout settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True, range=[-10, 10], title='X'),
            yaxis=dict(visible=True, range=[-10, 10], title='Y'),
            zaxis=dict(visible=True, range=[-10, 10], title='Z'),
            aspectmode='cube'
        ),
        title=f'Sun Position θ={theta_value}°, φ={phi_value}°, Cylinder Radius={cylinder_radius}',
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
