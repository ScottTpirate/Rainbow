import json
import numpy as np
import calculate  # Updated calculate.py with 3D functions

# Custom JSON encoder to handle NumPy data types
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.int_, int)):
            return int(o)
        elif isinstance(o, (np.floating, np.float_, float)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

# Parameters
radius = 1.0  # Should match the radius used in calculate.py
cylinder_radius = 0.5  # Adjust for the size of the incoming cylinder
num_rays = 100  # Adjust for resolution
n_air = calculate.n_air
refractive_indices = calculate.refractive_indices
sphere_center = np.array([0.0, 0.0, 0.0])

data = {}

# Define ranges for Sun's azimuth and elevation angles
azimuth_angles = np.arange(0, 361, 30)   # 0° to 360° in 30° increments
elevation_angles = np.arange(-90, 91, 30)  # -90° to 90° in 30° increments

# Loop over azimuth and elevation angles to simulate different Sun positions
for theta_value in azimuth_angles:
    for phi_value in elevation_angles:
        print(f"Processing θ={theta_value}°, φ={phi_value}°")
        data_entry = {}
        data_entry["rays"] = []

        # Convert angles to radians
        theta_rad = np.radians(theta_value)
        phi_rad = np.radians(phi_value)

        # Compute the incoming light direction vector D
        D = np.array([
            np.cos(phi_rad) * np.cos(theta_rad),
            np.cos(phi_rad) * np.sin(theta_rad),
            np.sin(phi_rad)
        ])
        D /= np.linalg.norm(D)

        # Compute orthonormal basis vectors perpendicular to D
        def compute_perpendicular_vectors(D):
            if abs(D[0]) < 0.9:
                U = np.array([1, 0, 0])
            else:
                U = np.array([0, 1, 0])
            V1 = U - np.dot(U, D) * D
            V1 /= np.linalg.norm(V1)
            V2 = np.cross(D, V1)
            return V1, V2

        V1, V2 = compute_perpendicular_vectors(D)

        # Generate rays on the surface of a cylinder
        theta_values = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        ray_origins = []
        r = cylinder_radius
        for theta in theta_values:
            offset = r * np.cos(theta) * V1 + r * np.sin(theta) * V2
            P0 = sphere_center - D * 5.0  # Start rays before the sphere
            origin = P0 + offset
            ray_origins.append(origin)

        # Collect rays
        rays = calculate.calculate_rays_cylinder(ray_origins, D, refractive_indices, radius)

        for ray in rays:
            color = ray['color']
            path = ray['path']
            ray_entry = {
                "color": color,
                "path": path
            }
            data_entry["rays"].append(ray_entry)

        # Store data for the current Sun position
        key = f"theta_{theta_value}_phi_{phi_value}"
        data[key] = data_entry

# Writing data to a JSON file
with open('rainbow_data_3D.json', 'w') as file:
    json.dump(data, file, cls=EnhancedJSONEncoder)

print("Data generation complete.")
