from flask import Flask, render_template, request, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Use this backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json
import traceback


app = Flask(__name__)

# Globally define the min and max values for your plot axes
minXValue = -1.5  # Replace with the actual min value of X from your data
maxXValue = 1.25   # Replace with the actual max value of X from your data
minYValue = -1.5  # Replace with the actual min value of Y from your data
maxYValue = 1.25  # Replace with the actual max value of Y from your data

# Load your JSON data
# with open('Rainbow/exaggerated_rainbow_data.json') as f:
#     data = json.load(f)

with open('rainbow_data.json') as f:
    data = json.load(f)

@app.route('/')
def index():
    # Render a template that contains your HTML and JS for the slider
    return render_template('index.html')

@app.route('/plot/<alpha>')
def plot(alpha):
    try:
        # Attempt to create the plot
        fig = create_plot_from_json(alpha)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        # Print full stack trace to console
        traceback.print_exc()
        # If an error occurs, print it to the console and return a 500 error
        print(e)
        return jsonify({"error": str(e)}), 500

def create_plot_from_json(alpha):

    # Access the data for the specific alpha
    alpha_key = f"alpha_{alpha}"
    if alpha_key not in data:
        raise ValueError(f"No data for alpha {alpha}")

    alpha_data = data[alpha_key]

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a circle patch
    circle = patches.Circle((0.5, 0.5), 0.5, color='b', fill=False)

    # Add the circle to the plot
    ax.add_patch(circle)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Iterate over each color in the data for the specific alpha
    for color, color_data in alpha_data.items():
        # Each color data contains xa, ya, xb, yb, etc.
        # Here you plot lines between (xa[0], ya[0]) and (xa[1], ya[1])
        if color == "incoming":
            ax.plot(color_data['xa'], color_data['ya'], color='gold')
        else:
            # and similarly for xb, yb; xc, yc; xd, yd
            ax.plot(color_data['xb'], color_data['yb'], color=color)
            ax.plot(color_data['xc'], color_data['yc'], color=color)
            ax.plot(color_data['xd'], color_data['yd'], color=color)

    # Set the axis limits if necessary
    ax.set_xlim([minXValue, maxXValue])
    ax.set_ylim([minYValue, maxYValue])

    # turn off the axis entirely
    ax.axis('off')

    # Optional: Configure other plot settings such as labels, title, legend, etc.
    ax.set_title(f"Visualization for alpha {alpha}")

    plt.close(fig)  # Close the figure before returning

    return fig

if __name__ == '__main__':
    app.run(debug=True)
