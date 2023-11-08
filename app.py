from flask import Flask, jsonify, render_template_string, request
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Simulated plotting function (replace with your actual plotting function)
def plot_rays_final(alpha_slider_value, radius):
    fig, ax = plt.subplots()
    # Your plotting code here...
    ax.set_title(f'Plot for angle={alpha_slider_value} and radius={radius}')
    plt.close(fig)
    return fig

# Convert plot to image
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template_string('''<!doctype html>
<html>
<head>
    <title>Interactive Plot</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script type="text/javascript">
    $(function() {
        function updatePlot() {
            var angle = $('#angleSlider').val();
            var radius = $('#radiusSlider').val();
            $.ajax({
                url: '/update_plot',
                data: { 'angle': angle, 'radius': radius },
                type: 'GET',
                success: function(response) {
                    $('#plotImage').attr('src', 'data:image/png;base64,' + response.image);
                },
                error: function(error) {
                    console.log(error);
                }
            });
        }
        $('#angleSlider, #radiusSlider').on('input change', updatePlot);
        updatePlot(); // Initial plot
    });
    </script>
</head>
<body>
    <h1>Interactive Plot with Flask</h1>
    Angle: <input type="range" id="angleSlider" min="0" max="80" value="45" step="1"><br>
    Radius: <input type="range" id="radiusSlider" min="0.1" max="1.0" value="0.4" step="0.01"><br>
    <img id="plotImage" src=""/>
</body>
</html>''')

@app.route('/update_plot')
def update_plot():
    angle = float(request.args.get('angle', 45))
    radius = float(request.args.get('radius', 0.4))
    fig = plot_rays_final(alpha_slider_value=angle, radius=radius)
    image = fig_to_base64(fig)
    return jsonify({'image': image})

if __name__ == '__main__':
    app.run(debug=True)
