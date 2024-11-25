# toJSON_3d.py

import json
import numpy as np
import calculate  # Ensure calculate.py is updated with 3D functions

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
radius = 1.0  # Sphere radius
cylinder_radii = np.arange(0.1, 1.1, 0.1)  # Cylinder radii from 0.1 to 1.0
num_rays = 100  # Adjust for resolution
n_air = calculate.n_air
refractive_indices = calculate.refractive_indices
sphere_center = np.array([0.0, 0.0, 0.0])

# Define ranges for Sun's azimuth and elevation angles
azimuth_angles = np.arange(0, 361, 30)    # 0° to 360° in 30° increments
elevation_angles = np.arange(-90, 91, 30)  # -90° to 90° in 30° increments

# Loop over azimuth angles
for theta_value in azimuth_angles:
    print(f"Processing θ={theta_value}°")
    data = {}  # Initialize data for this azimuth angle

    # Loop over elevation angles
    for phi_value in elevation_angles:
        print(f"  Processing φ={phi_value}°")
        data_entry = {}
        data_entry["rays"] = {}

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

        # Loop over cylinder radii
        for cylinder_radius in cylinder_radii:
            print(f"    Processing Cylinder Radius: {cylinder_radius}")
            rays_list = []

            # Generate rays on the surface of a cylinder
            theta_values_ray = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
            ray_origins = []
            r = cylinder_radius
            for theta in theta_values_ray:
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
                    "path": path  # path is a dict with segments and points
                }
                rays_list.append(ray_entry)

            # Store rays for the current cylinder radius
            cylinder_radius_key = f"radius_{cylinder_radius:.1f}"
            data_entry["rays"][cylinder_radius_key] = rays_list

        # Store data entry for the current elevation angle
        phi_key = f"phi_{phi_value}"
        data[phi_key] = data_entry

    # Writing data to a JSON file for the current azimuth angle
    filename = f'rainbow_data_theta_{theta_value}.json'
    with open(filename, 'w') as file:
        json.dump(data, file, cls=EnhancedJSONEncoder)

    print(f"Data generation complete for θ={theta_value}°")

print("All data generation complete.")
