import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Function to draw the angles
def draw_angles(angle):
    plt.clf()
    alpha = np.radians(angle)  # Convert to radians
    ior = 1.333  # index of refraction for water
    
    # Snell's Law at point A
    beta = np.arcsin(np.sin(alpha)/ior)
    
    # Internal Reflection angle at B
    gamma = np.pi - 2*beta
    
    # Snell's Law at point C
    delta = beta - alpha

    # Total deflection
    D_alpha = alpha + gamma + delta

    plt.plot([0, np.cos(alpha)], [0, np.sin(alpha)], label=f"Incident ray (angle = {angle} deg)")
    plt.plot([np.cos(alpha), np.cos(alpha) + np.cos(beta)], [np.sin(alpha), np.sin(alpha) + np.sin(beta)], label=f"Refracted ray at A (angle = {np.degrees(beta):.2f} deg)")
    plt.plot([np.cos(alpha) + np.cos(beta), np.cos(alpha) + np.cos(beta) - np.cos(gamma)], [np.sin(alpha) + np.sin(beta), np.sin(alpha) + np.sin(beta) - np.sin(gamma)], label=f"Reflected ray at B (angle = {np.degrees(gamma):.2f} deg)")
    plt.plot([np.cos(alpha) + np.cos(beta) - np.cos(gamma), np.cos(alpha) + np.cos(beta)], [np.sin(alpha) + np.sin(beta) - np.sin(gamma), np.sin(alpha) + np.sin(beta)], label=f"Refracted ray at C")
    plt.legend()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Total Deflection D_alpha = {np.degrees(D_alpha):.2f}")
    canvas.draw()

# Tkinter window setup
root = tk.Tk()
root.title("Rainbow Formation")

# Adding the slider
angle_slider = ttk.Scale(root, from_=0, to_=90, orient="horizontal", command=lambda val: draw_angles(float(val)))
angle_slider.set(45)
angle_slider.pack()

# Matplotlib setup
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

draw_angles(45)
root.mainloop()
