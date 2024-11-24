# calculate.py
import numpy as np
import matplotlib.pyplot as plt

# Define the center of the circle (water droplet)
center_x = 0.5
center_y = 0.5

# Refractive indices
n_air = 1.000293  # Air
n_water = 1.33    # Water

# Factor for extending normal lines (for plotting purposes)
factor = 0.5


def calculate_normal_vector(x, y):
    """
    Calculate the normal vector at a specific point on the circle.
    
    Parameters:
        x (float): x-coordinate of the point on the circle.
        y (float): y-coordinate of the point on the circle.
    
    Returns:
        normal_vector (np.ndarray): Normalized normal vector.
    """
    # Vector from circle center to point
    normal_vector = np.array([x - center_x, y - center_y])
    # Normalize the vector
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector


def calculate_refracted_ray(I, N, n1, n2):
    """
    Calculate the refracted ray vector using Snell's Law in vector form.
    
    Parameters:
        I (np.ndarray): Incident vector, normalized.
        N (np.ndarray): Normal vector at the point of incidence, normalized.
        n1 (float): Refractive index of the first medium.
        n2 (float): Refractive index of the second medium.
    
    Returns:
        T (np.ndarray or None): Refracted vector, normalized. None if total internal reflection occurs.
    """
    cos_theta_i = -np.dot(N, I)
    sin2_theta_t = (n1 / n2) ** 2 * (1 - cos_theta_i ** 2)
    if sin2_theta_t > 1.0:
        # Total internal reflection
        return None
    cos_theta_t = np.sqrt(1 - sin2_theta_t)
    T = (n1 / n2) * I + (n1 / n2 * cos_theta_i - cos_theta_t) * N
    T /= np.linalg.norm(T)
    return T


def calculate_reflected_ray(I, N):
    """
    Calculate the reflected ray vector.
    
    Parameters:
        I (np.ndarray): Incident vector, normalized.
        N (np.ndarray): Normal vector at the point of incidence, normalized.
    
    Returns:
        R (np.ndarray): Reflected vector, normalized.
    """
    R = I - 2 * np.dot(I, N) * N
    R /= np.linalg.norm(R)
    return R


def line_circle_intersection(P0, D, center, radius):
    """
    Find the intersection points of a line and a circle.
    
    Parameters:
        P0 (np.ndarray): Starting point of the line.
        D (np.ndarray): Direction vector of the line, normalized.
        center (np.ndarray): Center of the circle.
        radius (float): Radius of the circle.
    
    Returns:
        t_values (list): List of parameter t where the line intersects the circle.
    """
    # Coefficients of the quadratic equation At^2 + Bt + C = 0
    a = np.dot(D, D)
    b = 2 * np.dot(D, P0 - center)
    c = np.dot(P0 - center, P0 - center) - radius ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return []  # No intersection
    elif discriminant == 0:
        t = -b / (2 * a)
        return [t]
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        return [t1, t2]


def line_a(alpha, radius):
    """
    Calculate the incoming ray (line a).
    
    Parameters:
        alpha (float): Angle of incidence in degrees.
        radius (float): Radius of the circle.
    
    Returns:
        xa (np.ndarray): x-coordinates of the incoming ray.
        ya (np.ndarray): y-coordinates of the incoming ray.
    """
    # Intersection point with the circle
    x = center_x - radius * np.sin(np.radians(alpha))
    y = center_y + radius * np.cos(np.radians(alpha))

    # Incoming ray starts from far left (e.g., x = -5)
    xa = np.array([-5.0, x])
    ya = np.array([y, y])
    return xa, ya


def line_b(radius, n1, n2, xa, ya):
    """
    Calculate the refracted ray inside the circle (line b).
    
    Parameters:
        radius (float): Radius of the circle.
        n1 (float): Refractive index of the first medium (air).
        n2 (float): Refractive index of the second medium (water).
        xa (np.ndarray): x-coordinates of the incoming ray.
        ya (np.ndarray): y-coordinates of the incoming ray.
    
    Returns:
        xb (np.ndarray): x-coordinates of the refracted ray.
        yb (np.ndarray): y-coordinates of the refracted ray.
    """
    # Point of incidence
    P0 = np.array([xa[-1], ya[-1]])
    # Incident vector (normalized)
    I = np.array([xa[1] - xa[0], ya[1] - ya[0]])
    I /= np.linalg.norm(I)
    # Normal vector at point of incidence
    N = calculate_normal_vector(P0[0], P0[1])
    # Refracted vector
    T = calculate_refracted_ray(I, N, n1, n2)
    if T is None:
        return None, None  # Total internal reflection
    # Intersection with opposite side of circle
    t_values = line_circle_intersection(P0, T, np.array([center_x, center_y]), radius)
    # Choose the appropriate t (positive and not too small)
    t = max([t for t in t_values if t > 1e-6], default=None)
    if t is None:
        return None, None
    P_end = P0 + T * t
    xb = np.array([P0[0], P_end[0]])
    yb = np.array([P0[1], P_end[1]])
    return xb, yb


def line_c(radius, xb, yb):
    """
    Calculate the internally reflected ray inside the circle (line c).
    
    Parameters:
        radius (float): Radius of the circle.
        xb (np.ndarray): x-coordinates of the refracted ray inside the circle.
        yb (np.ndarray): y-coordinates of the refracted ray inside the circle.
    
    Returns:
        xc (np.ndarray): x-coordinates of the reflected ray.
        yc (np.ndarray): y-coordinates of the reflected ray.
    """
    # Point of reflection
    P0 = np.array([xb[-1], yb[-1]])
    # Incident vector (normalized)
    I = np.array([xb[-1] - xb[0], yb[-1] - yb[0]])
    I /= np.linalg.norm(I)
    # Normal vector at point of reflection
    N = calculate_normal_vector(P0[0], P0[1])
    # Reflected vector
    R = calculate_reflected_ray(I, N)
    # Intersection with circle
    t_values = line_circle_intersection(P0, R, np.array([center_x, center_y]), radius)
    # Choose the appropriate t (positive and not too small)
    t = max([t for t in t_values if t > 1e-6], default=None)
    if t is None:
        return None, None
    P_end = P0 + R * t
    xc = np.array([P0[0], P_end[0]])
    yc = np.array([P0[1], P_end[1]])
    return xc, yc


def line_d(radius, xc, yc, n1, n2):
    """
    Calculate the exiting refracted ray from the circle (line d).
    
    Parameters:
        radius (float): Radius of the circle.
        xc (np.ndarray): x-coordinates of the internally reflected ray.
        yc (np.ndarray): y-coordinates of the internally reflected ray.
        n1 (float): Refractive index inside the circle (water).
        n2 (float): Refractive index outside the circle (air).
    
    Returns:
        xd (np.ndarray): x-coordinates of the exiting ray.
        yd (np.ndarray): y-coordinates of the exiting ray.
    """
    # Point of incidence (where ray exits the circle)
    P0 = np.array([xc[-1], yc[-1]])
    # Incident vector (normalized)
    I = np.array([xc[-1] - xc[0], yc[-1] - yc[0]])
    I /= np.linalg.norm(I)
    # Normal vector at point of incidence (pointing outward)
    N = -calculate_normal_vector(P0[0], P0[1])
    # Refracted vector
    T = calculate_refracted_ray(I, N, n1, n2)
    if T is None:
        return None, None  # Total internal reflection
    # Extend the ray for visualization (e.g., extend by factor 10)
    P_end = P0 + T * 10
    xd = np.array([P0[0], P_end[0]])
    yd = np.array([P0[1], P_end[1]])
    return xd, yd


def plot_rays_final(alpha_slider_value, radius):
    """
    Plot the entire ray path through the droplet for a given incident angle.

    Parameters:
    - alpha_slider_value: Incident angle in degrees (0 to 60).
    - radius: Radius of the circle (droplet).
    """
    # Adjust angle to match geometry
    alpha = alpha_slider_value

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Draw the circle (droplet)
    circle = plt.Circle((center_x, center_y), radius, color='b', fill=False)
    ax2.add_patch(circle)

    # Line a (incoming ray)
    xa, ya = line_a(alpha, radius)
    ax2.plot(xa, ya, 'gold', label='Incoming Ray')

    # Define refractive indices for colors
    refractive_indices = {
        'red': 1.331,
        'orange': 1.332,
        'yellow': 1.333,
        'green': 1.335,
        'blue': 1.338,
        'violet': 1.342
    }

    # Plot rays for each color
    for color, n_water in refractive_indices.items():
        # Line b (refracted ray inside the droplet)
        xb, yb = line_b(radius, n_air, n_water, xa, ya)
        if xb is None:
            continue  # Skip if total internal reflection occurs
        ax2.plot(xb, yb, color=color)

        # Line c (reflected ray inside the droplet)
        xc, yc = line_c(radius, xb, yb)
        if xc is None:
            continue
        ax2.plot(xc, yc, color=color)

        # Line d (exiting ray)
        xd, yd = line_d(radius, xc, yc, n_water, n_air)
        if xd is None:
            continue
        ax1.plot(xd, yd, color=color, linewidth=2.5, alpha=0.7)
        ax2.plot(xd, yd, color=color)

    # Adjust axes
    ax1.set_xlim(-15, 2)
    ax1.set_ylim(-10, 2)
    ax1.set_title("Full View")

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_title("Zoomed View")

    # Set aspect ratio
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

    # Show legends
    ax2.legend()

    plt.show()



if __name__ == '__main__':
    # Example usage
    plot_rays_final(alpha_slider_value=45, radius=0.5)
    
    # Test the functions with an example angle and radius
    alpha_test = 45  # degrees
    radius_test = 0.5

    xa, ya = line_a(alpha_test, radius_test)
    xb, yb = line_b(radius_test, n_air, n_water, xa, ya)
    if xb is not None:
        xc, yc = line_c(radius_test, xb, yb)
        if xc is not None:
            xd, yd = line_d(radius_test, xc, yc, n_water, n_air)
            # Print results or plot if desired
            print("xa:", xa)
            print("ya:", ya)
            print("xb:", xb)
            print("yb:", yb)
            print("xc:", xc)
            print("yc:", yc)
            print("xd:", xd)
            print("yd:", yd)
