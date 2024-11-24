import numpy as np
import json

def compute_normal(point, sphere_center):
    normal = point - sphere_center
    normal /= np.linalg.norm(normal)
    return normal


def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(ray_direction, oc)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return None  # No intersection
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        return t1, t2  # Return both intersection points


def refract(incident, normal, n1, n2):
    mu = n1 / n2
    cos_theta_i = -np.dot(normal, incident)
    sin_theta_t2 = mu ** 2 * (1 - cos_theta_i ** 2)
    
    if sin_theta_t2 > 1:
        return None  # Total internal reflection
    cos_theta_t = np.sqrt(1 - sin_theta_t2)
    refracted = mu * incident + (mu * cos_theta_i - cos_theta_t) * normal
    refracted /= np.linalg.norm(refracted)
    return refracted



def reflect(incident, normal):
    reflected = incident - 2 * np.dot(incident, normal) * normal
    reflected /= np.linalg.norm(reflected)
    return reflected


def trace_ray_full(ray_origin, ray_direction, sphere_center, sphere_radius, n_air, n_water):
    # First intersection
    intersections = ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius)
    if intersections is None:
        return None
    t1, t2 = intersections
    t_enter = min(t1, t2)
    point_enter = ray_origin + t_enter * ray_direction
    normal_enter = compute_normal(point_enter, sphere_center)

    # Refraction at entry
    refracted_dir = refract(ray_direction, normal_enter, n_air, n_water)
    if refracted_dir is None:
        return None  # Total internal reflection

    # Second intersection inside sphere
    intersections_inside = ray_sphere_intersect(point_enter, refracted_dir, sphere_center, sphere_radius)
    t3, t4 = intersections_inside
    t_inside = max(t3, t4)
    point_inside = point_enter + t_inside * refracted_dir
    normal_inside = compute_normal(point_inside, sphere_center)

    # Reflection inside sphere
    reflected_dir = reflect(refracted_dir, normal_inside)

    # Exit point
    intersections_exit = ray_sphere_intersect(point_inside, reflected_dir, sphere_center, sphere_radius)
    t5, t6 = intersections_exit
    t_exit = max(t5, t6)
    point_exit = point_inside + t_exit * reflected_dir
    normal_exit = compute_normal(point_exit, sphere_center)

    # Refraction at exit
    exit_dir = refract(reflected_dir, normal_exit, n_water, n_air)
    if exit_dir is None:
        return None  # Total internal reflection

    # Compile the full path
    # path = {
    #     'points': [
    #         ray_origin,
    #         point_enter,
    #         point_inside,
    #         point_exit
    #     ],
    #     'directions': [
    #         ray_direction,
    #         refracted_dir,
    #         reflected_dir,
    #         exit_dir
    #     ]
    # }

    # Instead of returning numpy arrays, convert data to lists for JSON serialization
    path = {
        'points': [
            point.tolist() for point in [ray_origin, point_enter, point_inside, point_exit]
        ],
        'directions': [
            direction.tolist() for direction in [ray_direction, refracted_dir, reflected_dir, exit_dir]
        ]
    }

    return path


def collect_and_save_ray_data():
    ray_data = {}
    sphere_center = np.array([0, 0, 0], dtype=float)
    sphere_radius = 1.0
    n_air = 1.0
    refractive_indices = {
        'red': 1.331,
        'green': 1.333,
        'blue': 1.337
    }

    # Entry levels (adjust as needed)
    levels = np.linspace(-0.9, 0.9, 10)  # Avoid grazing angles

    for level in levels:
        ray_origin = np.array([-5, level, 0], dtype=float)
        ray_direction = np.array([1, 0, 0], dtype=float)  # Direction towards the droplet
        ray_direction /= np.linalg.norm(ray_direction)

        ray_data_level = {}

        # Loop over different colors
        for color, n_water in refractive_indices.items():
            path = trace_ray_full(ray_origin, ray_direction, sphere_center, sphere_radius, n_air, n_water)
            if path:
                ray_data_level[color] = path

        ray_data[str(level)] = ray_data_level

    # Write to JSON file
    with open('ray_data.json', 'w') as json_file:
        json.dump(ray_data, json_file, indent=4)


collect_and_save_ray_data()
