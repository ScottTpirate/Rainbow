<p align="center">
    <img src="fun_rainbow_graphic.png" width="200" height="200">
</p>

<h1 align="center">Rainbow Visualization Tool</h1>
<h3 align="center">Somewhere Within the Rainbow</h3>



## Introduction
This project presents an interactive tool to visualize and understand the mechanics of a rainbow, specifically focusing on how light refracts, reflects, and refracts again to create this natural phenomenon. The tool leverages several datasets to provide a dynamic and interactive plotting experience.

## Motivation
The motivation behind this project is to demystify the complex process of rainbow formation through an engaging and educational tool. This tool aims to make the scientific principles behind rainbows accessible to a broader audience, regardless of their background in physics or meteorology.

## Libraries Used
- Flask
- Matplotlib
- JSON
- IO
- Traceback
- OS

## Files in the Repository
- `app.py`: The main Flask application file containing the backend logic for the interactive plots.
- `exaggerated_rainbow_data.json`: A dataset providing exaggerated representations of rainbow light paths.
- `exaggerated_rainbow_data_SMALL.json`: A smaller version of the exaggerated rainbow dataset.
- `rainbow_data_SMALL.json`: A smaller dataset for basic rainbow data.
- `rainbow_data.json`: The primary dataset containing detailed data points of rainbow formation.
- `templates/index.html`: HTML template for the web interface.
- `static/`: Folder containing CSS and JavaScript files for the web interface.

## Analysis Summary
The interactive plots generated by this tool offer a detailed visualization of how different wavelengths of light bend and reflect within water droplets to create a rainbow. Users can adjust parameters to see how these changes affect the appearance and intensity of the rainbow.

## Acknowledgements
This project was inspired by the Stack Overflow Developer Survey, and draws upon a variety of resources including scientific literature on meteorology and optics, as well as community discussions on platforms like StackOverflow and Kaggle.

## How to Run
To run this tool locally:
1. Ensure you have Python and the necessary libraries installed.
2. Clone this repository to your local machine.
3. Navigate to the cloned directory and run `python app.py` in your terminal.
4. Open your web browser and go to `http://localhost:5000`.

---

