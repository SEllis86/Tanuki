import os
import base64
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from tifffile import imread
from functools import lru_cache
from threading import Timer
import webbrowser

def run_dash_interactive_viewer(df, default_color_col, color_map=None, legend_order=None, master_dir=".", port=8050):
    app = Dash(__name__)

    @lru_cache(maxsize=10)
    def get_cached_data(subdir):
        premsi = os.path.join(master_dir, subdir, "MovedMSI_cropped_multichannel.tif")
        seg = os.path.join(master_dir, subdir, "PREMSI_seg.npy")
        if not os.path.exists(premsi) or not os.path.exists(seg): return None, None
        img = imread(premsi).astype(np.float32)
        if img.ndim == 3 and img.shape[0] <= 100: img = np.moveaxis(img, 0, -1)
        if np.max(img) > 0: img /= np.max(img)
        mask = np.load(seg, allow_pickle=True).item()['masks']
        return img, mask

    app.layout = html.Div([
        dcc.Dropdown(id='color-dropdown', options=[{'label': c, 'value': c} for c in df.columns], value=default_color_col),
        dcc.Graph(id="projection-plot", style={"height": "800px"}),
        html.Img(id="cell-image"), html.Img(id="mask-overlay-image"), dcc.Graph(id="spectra-plot")
    ])

    @app.callback(Output("projection-plot", "figure"), [Input("color-dropdown", "value")])
    def update_scatter(color_choice):
        fig = px.scatter(df, x="AlignedMass1", y="AlignedMass2", color=color_choice, color_discrete_map=color_map, category_orders={color_choice: legend_order}, hover_name="sample name")
        return fig

    # Add back the @app.callback for hovering over cells here from your original script
    
    Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(port=port, debug=False)
