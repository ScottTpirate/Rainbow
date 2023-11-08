import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import math
import sympy as sp

center_x = 0.5  
center_y = 0.5

# Air has an index of refraction of about 1.0, and water about 1.33
n_air = 1.0
n_water = 1.33

factor = 0.5



def reflect_line_of_equal_length(incoming_vector, incidence_point, start_point):

    # Calculate the length of the original line
    end_point = incidence_point
    original_line_vector = sp.Matrix(end_point) - sp.Matrix(start_point)
    length_of_original_line = original_line_vector.norm()
    
    # Calculate the normalized normal vector at the point of incidence
    normal_vector = (sp.Matrix(incidence_point) - sp.Matrix([center_x, center_y])).normalized()
    
    # Reflect the incoming vector across the normal vector
    reflected_vector = sp.Matrix(incoming_vector) - 2 * (sp.Matrix(incoming_vector).dot(normal_vector)) * normal_vector
    
    # Normalize the reflected vector and scale it to the length of the original line
    normalized_reflected_vector = reflected_vector.normalized()
    scaled_reflected_vector = normalized_reflected_vector * length_of_original_line
    
    # Calculate the end point of the reflected line
    reflected_line_end_point = sp.Matrix(incidence_point) + scaled_reflected_vector
    
    return reflected_line_end_point


def calculate_refracted_ray(incidence_point, normal_vector, incoming_vector, n1, n2):
    # Normalize vectors
    normal_vector = sp.Matrix(normal_vector).normalized()
    incoming_vector = sp.Matrix(incoming_vector).normalized()
    
    # Calculate the angle of incidence using the dot product
    angle_of_incidence = sp.acos(incoming_vector.dot(normal_vector))
    
    # Apply Snell's Law to find the sine of the angle of refraction
    sin_angle_of_refraction = n1 / n2 * sp.sin(angle_of_incidence)
    
    # Check for total internal reflection
    if abs(sin_angle_of_refraction) > 1:
        raise ValueError("Total internal reflection: the angle of refraction cannot be computed.")
    
    # Calculate the actual angle of refraction
    angle_of_refraction = sp.asin(sin_angle_of_refraction)
    
    # Determine the direction of the cross product
    cross_product = normal_vector[0] * incoming_vector[1] - normal_vector[1] * incoming_vector[0]
    
    # If the cross product is negative, the incoming vector is on the left side of the normal and we need to rotate clockwise
    # If it's positive, the incoming vector is on the right side and we need to rotate counterclockwise
    if cross_product < 0:
        rotation_matrix = sp.Matrix([[sp.cos(angle_of_refraction), sp.sin(angle_of_refraction)],
                                     [-sp.sin(angle_of_refraction), sp.cos(angle_of_refraction)]])
    else:
        rotation_matrix = sp.Matrix([[sp.cos(-angle_of_refraction), sp.sin(-angle_of_refraction)],
                                     [-sp.sin(-angle_of_refraction), sp.cos(-angle_of_refraction)]])
    
    # Calculate the refracted ray vector
    refracted_vector = rotation_matrix * normal_vector
    
    # Assuming the refracted ray should have the same length as the incoming vector
    length_of_incoming_ray = sp.Matrix(incoming_vector).norm()
    refracted_ray_end_point = sp.Matrix(incidence_point) + refracted_vector * length_of_incoming_ray
    
    return refracted_ray_end_point.evalf()


def calculate_point_b(a_x, a_y, angle_a, normal_vector):
    # Convert angle to radians for computation
    angle_a_radians = sp.rad(angle_a)
    
    # Calculate the vector from center to A
    vector_ca = sp.Matrix([a_x - center_x, a_y - center_y])
    
    # Normalize the normal vector
    normal_vector_normalized = sp.Matrix(normal_vector).normalized()
    
    # Rotate the vector CA by the angle at A to get the vector CB
    rotation_matrix = sp.Matrix([[sp.cos(angle_a_radians), -sp.sin(angle_a_radians)],
                                 [sp.sin(angle_a_radians), sp.cos(angle_a_radians)]])
    vector_cb = rotation_matrix * vector_ca
    
    # Since the triangle is isosceles and angles at A and B are equal, the lengths |CA| and |CB| are equal
    # We use the normal vector to determine the direction from the center to B
    direction_to_b = sp.sign((vector_cb.T * normal_vector_normalized)[0])
    
    # Compute the coordinates of B by translating the center by the vector CB
    b_x = center_x + direction_to_b * vector_cb[0]
    b_y = center_y - direction_to_b * vector_cb[1]

    evaluated_b_x = b_x.evalf()
    evaluated_b_y = b_y.evalf()
    
    return (evaluated_b_x, evaluated_b_y)


def plot_tangent_line_at_point(incidence_point, line_length=.3):
    # Compute the slope of the line from the center to the point (radius)
    delta_x = incidence_point[0] - center_x
    delta_y = incidence_point[1] - center_y
    
    # Check for vertical line to avoid division by zero
    if delta_x == 0:
        tangent_slope = float('inf')
    else:
        tangent_slope = -delta_x / delta_y

    # Define the line length for the tangent line segment
    delta_line = line_length / 2

    # Plotting the tangent line
    if tangent_slope == float('inf'):  # Vertical line
        x_vals = [incidence_point[0], incidence_point[0]]
        y_vals = [incidence_point[1] - delta_line, incidence_point[1] + delta_line]
    else:
        x_vals = [incidence_point[0] - delta_line, incidence_point[0] + delta_line]
        y_vals = [incidence_point[1] - delta_line * tangent_slope, incidence_point[1] + delta_line * tangent_slope]
    
    plt.plot(x_vals, y_vals, 'k:', label='Tangent Line')


def calculate_refraction_angle(incidence_degrees, n1, n2):
    # Convert the incidence angle from degrees to radians
    incidence_radians = math.radians(incidence_degrees)
    
    # Use Snell's Law to find the sine of the refraction angle
    sin_refraction_angle = math.sin(incidence_radians) * n1 / n2
    
    # Check for total internal reflection
    if abs(sin_refraction_angle) > 1:
        raise ValueError("Total internal reflection occurs, no refraction possible.")
    
    # Find the refraction angle in radians using the arcsine function
    refraction_radians = math.asin(sin_refraction_angle)
    
    # Return the refraction angle in radians
    return refraction_radians


def calculate_normal_vector(intersect_x, intersect_y):
    """
    Calculate the normal vector at a specific point on the circle.

    :param intersect_x: x-coordinate of the intersection point
    :param intersect_y: y-coordinate of the intersection point
    :return: The normal vector as a tuple (nx, ny)
    """
    
    # Assuming center_x and center_y are the global coordinates of the circle's center
    global center_x
    global center_y

    # Calculate the vector from the center of the circle to the intersection point
    normal_dx = intersect_x - center_x
    normal_dy = intersect_y - center_y

    # Normalize the vector
    magnitude = math.sqrt(normal_dx**2 + normal_dy**2)
    normal_dx /= magnitude
    normal_dy /= magnitude

    return (normal_dx, normal_dy)


def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors.

    :param v1: The first vector as a tuple (x1, y1)
    :param v2: The second vector as a tuple (x2, y2)
    :return: The angle in degrees between the two vectors
    """
    
    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Calculate magnitudes
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Make sure cosine value is within -1 to 1 range to avoid numerical issues
    cos_angle = max(min(cos_angle, 1), -1)
    
    # Calculate the angle in radians and then convert to degrees
    angle = math.acos(cos_angle)
    angle_degrees = math.degrees(angle)
    
    return angle_degrees


def line_a(alpha, radius):
    # Compute intersection points
    x = 0.5 - radius * np.sin(np.deg2rad(alpha))
    y = 0.5 + radius * np.cos(np.deg2rad(alpha))
    
    return [-0.5, x], [y, y]


def line_b(alpha, radius):
    xa, ya = line_a(alpha, radius)
    x_intersect_a, y_intersect_a = xa[-1], ya[-1]
    
    vn = calculate_normal_vector(x_intersect_a, y_intersect_a)
    vn_positive = tuple(abs(x) for x in vn)

    # plot the vector in the negative (going out away from the center) but calculate it in the pos. maybe?
    # Extend the normal line by the specified factor
    extended_x = x_intersect_a + vn[0] * factor
    extended_y = y_intersect_a + vn[1] * factor

    # Plot the extended normal line
    plt.plot([center_x, extended_x], [center_y, extended_y], 'k:', label='Normal')

    #line a vector:
    da = (xa[1] - xa[0], ya[1] - ya[0])

    incidence_degrees = angle_between_vectors(vn_positive, da)

    refraction_radians = calculate_refraction_angle(incidence_degrees, n_air, n_water)
    refraction_degrees = math.degrees(refraction_radians)

    (B_x, B_y) = calculate_point_b(x_intersect_a, y_intersect_a, refraction_degrees, vn_positive)

    # Plot the refracted ray
    plt.plot([x_intersect_a, B_x], [y_intersect_a, B_y], 'r-', label='Refracted ray')

    return [x_intersect_a, B_x], [y_intersect_a, B_y]


def line_c(xb, yb):
    # Assuming we already have the end point of line b which is the intersection point
    x_intersect_b, y_intersect_b = xb[-1], yb[-1]

    vn = calculate_normal_vector(x_intersect_b, y_intersect_b)
    # vn_positive = tuple(abs(x) for x in vn)

    # plot the vector in the negative (going out away from the center) but calculate it in the pos. maybe?
    # Extend the normal line by the specified factor
    extended_x = x_intersect_b + vn[0] * factor
    extended_y = y_intersect_b + vn[1] * factor

    # Plot the extended normal line
    plt.plot([center_x, extended_x], [center_y, extended_y], 'k:', label='Normal_B')

    # line b vector:
    db = (xb[1] - xb[0], yb[1] - yb[0])

    # incidence_degrees = angle_between_vectors(vn_positive, db)

    (B_x, B_y) = reflect_line_of_equal_length(db, (x_intersect_b, y_intersect_b), (xb[0], yb[0]))
    
    return [x_intersect_b, B_x], [y_intersect_b, B_y]


def line_d(xc, yc):
    # Assuming we already have the end point of line c which is the intersection point
    x_intersect_c, y_intersect_c = xc[1], yc[1]

    vn = calculate_normal_vector(x_intersect_c, y_intersect_c)
    vn_positive = tuple(abs(x) for x in vn)

    extended_x_C = x_intersect_c + vn[0] * factor
    extended_y_C = y_intersect_c + vn[1] * factor

    plt.plot([center_x, extended_x_C], [center_y, extended_y_C], 'k:', label='Normal_C')

    #line c vector:
    dc = (xc[1] - xc[0], yc[1] - yc[0])

    incidence_degrees = angle_between_vectors(vn_positive, dc)

    refraction_radians = calculate_refraction_angle(incidence_degrees, n_water, n_air)
    
    refraction_degrees = math.degrees(refraction_radians)
    
    (x_end, y_end) = calculate_refracted_ray((x_intersect_c, y_intersect_c,), vn, dc, n_water, n_air)
    
    return [x_intersect_c, x_end], [y_intersect_c, y_end]


def plot_rays_final(alpha_slider_value, radius):
    alpha = 90 - alpha_slider_value
    # alpha = alpha_slider_value

    plt.figure(figsize=(8,8))
    circle = plt.Circle((center_x, center_y), radius, color='b', fill=False)
    plt.gca().add_patch(circle)

    plt.plot([center_x - radius, center_x + radius], [center_y, center_y], 'k:', label='Equator')
    
    # Line a
    xa, ya = line_a(alpha, radius)
    plt.plot(xa, ya, 'g-', label='Incoming ray')

    
    # Line b
    xb, yb = line_b(alpha, radius)
    # plt.plot(xb, yb, 'r-', label='Refracted ray')

    # Line c
    xc, yc = line_c(xb, yb)
    plt.plot(xc, yc, 'b-', label='Internally reflected ray')

    # Line d
    xd, yd = line_d(xc, yc)
    plt.plot(xd, yd, 'y-', label='Outgoing ray')

    # Uncomment to plot tangent lines
    # plot_tangent_line_at_point((xa[-1], ya[-1]))
    # plot_tangent_line_at_point((xb[-1], yb[-1]))
    # plot_tangent_line_at_point((xc[-1], yc[-1]))
    
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()



widgets.interactive(plot_rays_final, alpha_slider_value=widgets.FloatSlider(value=45, min=0, max=80, step=1, description='Angle:'), radius=widgets.FloatSlider(value=0.4, min=0.1, max=1.0, step=0.01, description='Radius:'))

