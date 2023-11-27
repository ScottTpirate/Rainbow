from flask import Flask, render_template, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Use this backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import json
import traceback
import os

app = Flask(__name__)

# Globally define the min and max values for your plot axes
minXValue = -1.5  # Replace with the actual min value of X from your data
maxXValue = 1.25   # Replace with the actual max value of X from your data
minYValue = -1.5  # Replace with the actual min value of Y from your data
maxYValue = 1.25  # Replace with the actual max value of Y from your data

# Load your JSON data
with open('exaggerated_rainbow_data.json') as f:
    data1 = json.load(f)

with open('exaggerated_rainbow_data_SMALL.json') as f:
    data2 = json.load(f)

with open('rainbow_data_SMALL.json') as f:
    data3 = json.load(f)

with open('rainbow_data.json') as f:
    data4 = json.load(f)


@app.route('/')
def index():
    # Render a template that contains your HTML and JS for the slider
    return render_template('index.html')

@app.route('/plot/<plot_type>/<alpha>')
def plot(plot_type, alpha):
    print("plot_type",plot_type)
    print("alpha",alpha)
    try:
        # Attempt to create the plot
        if plot_type == "plot1":
            fig = create_plot_from_json_iter(alpha)
        elif plot_type == "plot2":
            fig = create_plot_from_json_iter_exagg(alpha)
        elif plot_type == "plot3":
            fig = create_plot_from_json_all_exagg()
        else:
            fig = create_plot_from_json_all()

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

def create_plot_from_json_iter(alpha):

    # Access the data for the specific alpha
    alpha_key = f"alpha_{alpha}"
    if alpha_key not in data4:
        raise ValueError(f"No data for alpha {alpha}")

    alpha_data = data4[alpha_key]

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


def create_plot_from_json_iter_exagg(alpha):

    # Access the data for the specific alpha
    alpha_key = f"alpha_{alpha}"
    if alpha_key not in data1:
        raise ValueError(f"No data for alpha {alpha}")

    alpha_data = data1[alpha_key]

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



def create_plot_from_json_all_exagg():

    # Access the data for the specific alpha
    # alpha_key = f"alpha_{alpha}"
    # if alpha_key not in data:
    #     raise ValueError(f"No data for alpha {alpha}")

    

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a circle patch
    circle = patches.Circle((0.5, 0.5), 0.5, color='b', fill=False)

    # Add the circle to the plot
    ax.add_patch(circle)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    for alpha, alpha_data in data2.items():
    # alpha_data = data[alpha_key]

        # Iterate over each color in the data for the specific alpha
        for color, color_data in alpha_data.items():
            # Each color data contains xa, ya, xb, yb, etc.
            # Here you plot lines between (xa[0], ya[0]) and (xa[1], ya[1])
            if color == "incoming":
                ax.plot(color_data['xa'], color_data['ya'], color='gold')
            else:
                # and similarly for xb, yb; xc, yc; xd, yd

                # Find the position of the period
                # period_index = alpha.find('.')
                # first_two_digits = alpha[6:period_index]

                ax.plot(color_data['xb'], color_data['yb'], color=color, alpha=0.3)
                ax.plot(color_data['xc'], color_data['yc'], color=color, alpha=0.3)
                ax.plot(color_data['xd'], color_data['yd'], color=color, alpha=0.3)



    # Set the axis limits if necessary
    ax.set_xlim([minXValue, maxXValue])
    ax.set_ylim([minYValue, maxYValue])

    # turn off the axis entirely
    ax.axis('off')

    # Optional: Configure other plot settings such as labels, title, legend, etc.
    ax.set_title(f"Visualization for alpha {alpha}")

    plt.close(fig)  # Close the figure before returning

    return fig


def create_plot_from_json_all():

    # Access the data for the specific alpha
    # alpha_key = f"alpha_{alpha}"
    # if alpha_key not in data:
    #     raise ValueError(f"No data for alpha {alpha}")

    

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a circle patch
    circle = patches.Circle((0.5, 0.5), 0.5, color='b', fill=False)

    # Add the circle to the plot
    ax.add_patch(circle)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    for alpha, alpha_data in data3.items():
    # alpha_data = data[alpha_key]

        # Iterate over each color in the data for the specific alpha
        for color, color_data in alpha_data.items():
            # Each color data contains xa, ya, xb, yb, etc.
            # Here you plot lines between (xa[0], ya[0]) and (xa[1], ya[1])
            if color == "incoming":
                ax.plot(color_data['xa'], color_data['ya'], color='gold')
            else:
                # and similarly for xb, yb; xc, yc; xd, yd

                # Find the position of the period
                # period_index = alpha.find('.')
                # first_two_digits = alpha[6:period_index]

                ax.plot(color_data['xb'], color_data['yb'], color=color, alpha=0.3)
                ax.plot(color_data['xc'], color_data['yc'], color=color, alpha=0.3)
                ax.plot(color_data['xd'], color_data['yd'], color=color, alpha=0.3)



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
    port = int(os.environ.get('PORT', 5000))  # Use PORT if it's there, otherwise default to 5000
    app.run(host='0.0.0.0', port=port)
