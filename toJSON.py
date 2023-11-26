import json
from . import main
import numpy as np
import sympy

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, sympy.Float):
                return float(o)
            else:
                return super().default(o)
        except TypeError as te:
            print(f"TypeError: {te}")
            print(f"Unserializable object type: {type(o)} - {repr(o)}")
            raise


# Parameters
radius = 0.5 # your radius value
# refractive_indices = {
#     'red': 1.31,
#     'orange': 1.32,
#     'yellow': 1.33,
#     'green': 1.35,
#     'blue': 1.38,
#     'violet': 1.42
# }
refractive_indices = {
    'red': 1.331,
    'orange': 1.332,
    'yellow': 1.333,
    'green': 1.335,
    'blue': 1.338,
    'violet': 1.342
}

data = {}

for alpha in range(1, 61):  # Looping from 1 to 60
    data_entry = {}
    alpha_idk = 90 - alpha
    print("alpha_idk:",alpha_idk)
    xa, ya = main.line_a(alpha_idk, radius)
    data_entry["incoming"] = {'xa': xa, 'ya': ya}

    for color, n_water in refractive_indices.items():
        xb, yb = main.line_b(radius, main.n_air, n_water, xa, ya)
        xc, yc = main.line_c(xb, yb)
        xd, yd = main.line_d(xc, yc, n_water, main.n_air)
        print(f"Alpha:{alpha} | {color} | {n_water} \n")

        # Assuming you want to store the coordinates of lines b, c, and d
        data_entry[color] = {
            'xb': xb, 'yb': yb,
            'xc': xc, 'yc': yc,
            'xd': xd, 'yd': yd
        }
        

    data[f'alpha_{alpha}'] = data_entry


# Writing data to a JSON file
with open('Rainbow/rainbow_data.json', 'w') as file:
    json.dump(data, file, cls=EnhancedJSONEncoder)
