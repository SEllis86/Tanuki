"""Interactive Dash viewer (v5: defined inside the __main__ block)."""
from __future__ import annotations

import base64
import functools
import socket
import webbrowser
from pathlib import Path
from threading import Timer

import cv2
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, callback_context, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from . import analysis as _analysis
from .config import FEATURE_LABEL
from .utils import group_color_map, group_sort_key


def make_plot_frame(final_df_avg, feature_cols, raw_intensities_df):
    """The analysed cells with their RAW intensities restored, for the viewer (v5 main block)."""
    df_plot = final_df_avg.copy()

    # notes: Restore the raw values into df_plot strictly for Dash app rendering
    df_plot[feature_cols] = raw_intensities_df[feature_cols]

    df_plot["Status"] = df_plot["Individual Sample Type"].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) else "Unknown")
    df_plot["Value"] = df_plot["Individual Sample Type"].apply(
        lambda x: x.split('_')[1] if isinstance(x, str) and len(x.split('_')) > 1 else "Unknown")
    df_plot["StatusValue"] = list(zip(df_plot["Status"], df_plot["Value"]))

    return df_plot


def find_free_port(start_port):
    port_to_use = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port_to_use)) != 0:
                return port_to_use
            port_to_use += 1


def run_dash_interactive_viewer(df, default_color_col, color_map=None, symbol_col=None, symbol_map=None,
                                legend_order=None, master_dir=".", port=8050,
                                bonsai_segments=None):
    app = Dash(__name__)

    # v5: a well's masks are loaded once, not on every click (keyed on the file's mtime,
    # so a re-segmented well is picked up)
    @functools.lru_cache(maxsize=8)
    def _load_masks_cached(path_str, _mtime):
        seg_data = np.load(path_str, allow_pickle=True).item()
        return seg_data["masks"] if isinstance(seg_data, dict) and "masks" in seg_data else seg_data

    bonsai_edge_trace = None
    if bonsai_segments:
        _ex, _ey = [], []
        for x0, y0, x1, y1 in bonsai_segments:
            _ex.extend([x0, x1, None])
            _ey.extend([y0, y1, None])
        bonsai_edge_trace = go.Scatter(x=_ex, y=_ey, mode='lines',
                                       line=dict(color='rgba(120,120,120,0.45)', width=1),
                                       hoverinfo='skip', showlegend=False, name='tree')

    ignore_num_cols = {'mask_id', 'area_pixels', 'perimeter', 'extent', 'eccentricity',
                       'major_axis_length', 'minor_axis_length', 'solidity', 'branch_points',
                       'number_trunks', 'number_non_trunk_branches', 'number_branch_ends',
                       'total_object_skeleton_length', 'centroid_row', 'centroid_col',
                       'ClusterUnifiedDens', 'KmeansClusterUMAP',
                       'KmeansClusterdensUMAP', 'KmeansClusterTSNE', 'KmeansClusterPCA',
                       'ClusterNMF', 'ClusterLeiden',
                       'well_cells_segmented', 'well_cells_after_area_gate',
                       'well_cells_final', 'well_pct_retained'}
    mz_cols = []
    for c in df.columns:
        if c in ignore_num_cols:
            continue
        c_str = str(c)
        if c_str.startswith("mz_"):
            mz_cols.append(c)
        else:
            try:
                float(c_str)
                mz_cols.append(c)
            except ValueError:
                pass

    sorted_mz_cols = sorted(mz_cols, key=lambda x: float(str(x).replace('mz_', '')))
    mz_options = [{'label': f"m/z {str(c).replace('mz_', '')}", 'value': c} for c in sorted_mz_cols]
    # v3: Slide and CellType added; area_pixels too, for picking area caps from the plot
    _extra_colour_cols = [c for c in ["Batch", "Slide", "CellType", "Well", "Stim", "Sample Type + Stim",
                                      "mask_source", "area_pixels"] if c in df.columns]
    dropdown_options = [{'label': c, 'value': c} for c in
                        [default_color_col] + _extra_colour_cols + [col for col in df.columns if
                                                                    'Cluster' in col]] + mz_options

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Label("Projection: ", style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='projection-dropdown',
                             options=[
                                 {'label': f'{_analysis.COMBAT_LABEL} DensMAP ({FEATURE_LABEL})', 'value': 'DensMAP_Corrected'},
                                 {'label': f'Uncorrected DensMAP ({FEATURE_LABEL})', 'value': 'DensMAP_Uncorrected'},
                             ] + ([{'label': 'Uncorrected DensMAP (mass only)',
                                    'value': 'DensMAP_UncorrMassOnly'}]
                                  if 'DensMAP_UncorrMassOnly1' in df.columns else []) + (
                                 [{'label': 'DensMAP (morphology only)',
                                   'value': 'DensMAP_MorphOnly'}]
                                  if 'DensMAP_MorphOnly1' in df.columns else []) + [
                                 {'label': 'Standard UMAP (Mass)', 'value': 'UMAP'},
                                 {'label': 'Standard t-SNE (Mass)', 'value': 'TSNE'},
                             ] + ([{'label': 'Bonsai tree (Mass)', 'value': 'Bonsai'}]
                                  if 'Bonsai1' in df.columns else []) + [
                             ],
                             value='DensMAP_Corrected', clearable=False, style={'width': '350px'})
            ], style={'display': 'inline-block', 'marginRight': '40px', 'verticalAlign': 'top'}),

            # --- NEW: Color Mode Toggle and Dynamic Controls ---
            html.Div([
                html.Div([
                    html.Label("Coloring Mode: ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.RadioItems(
                        id='color-mode-toggle',
                        options=[
                            {'label': ' Standard', 'value': 'standard'},
                            {'label': ' m/z Ratio Calculator', 'value': 'ratio'}
                        ],
                        value='standard',
                        inline=True
                    )
                ], style={'marginBottom': '10px'}),

                html.Div(id='standard-color-div', children=[
                    html.Label("Color by: ", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='color-dropdown',
                                 options=dropdown_options,
                                 value=default_color_col, clearable=False, style={'width': '250px'})
                ]),

                # One numerator over up to THREE denominators:
                #     ratio = num / (num + den1 + den2 + den3)
                # Denominators 2 and 3 are clearable, so leaving them blank falls back to a
                # plain two- or three-mass ratio. The ratio stays bounded 0..1 either way.
                html.Div(id='ratio-color-div', style={'display': 'none'}, children=[
                    html.Div([
                        html.Label("Numerator m/z: ", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                        dcc.Dropdown(id='ratio-num', options=mz_options, value=sorted_mz_cols[0], clearable=False,
                                     style={'width': '180px'})
                    ], style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.Div([
                        html.Label("Denominator 1 m/z: ", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                        dcc.Dropdown(id='ratio-den', options=mz_options,
                                     value=sorted_mz_cols[1] if len(sorted_mz_cols) > 1 else sorted_mz_cols[0],
                                     clearable=False, style={'width': '180px'})
                    ], style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.Div([
                        html.Label("Denominator 2 m/z: ", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                        dcc.Dropdown(id='ratio-den2', options=mz_options,
                                     value=sorted_mz_cols[2] if len(sorted_mz_cols) > 2 else None,
                                     clearable=True, placeholder='(optional)', style={'width': '180px'})
                    ], style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.Div([
                        html.Label("Denominator 3 m/z: ", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                        dcc.Dropdown(id='ratio-den3', options=mz_options,
                                     value=sorted_mz_cols[3] if len(sorted_mz_cols) > 3 else None,
                                     clearable=True, placeholder='(optional)', style={'width': '180px'})
                    ], style={'display': 'inline-block'})
                ])
            ], style={'display': 'inline-block', 'verticalAlign': 'top'})
            # ---------------------------------------------------

        ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderBottom': '1px solid #ddd'}),

        html.Div([
            dcc.Graph(id="projection-plot", style={"height": "800px", "width": "55%"}),

            html.Div([
                html.H4(id="cell-inspection-title", children="Cell Inspection",
                        style={'textAlign': 'center', 'marginBottom': '5px'}),

                dcc.Checklist(
                    id='interaction-checkboxes',
                    options=[
                        {'label': ' Lock Selection to Click (Disable Hover)', 'value': 'click'},
                        {'label': ' Generate Live m/z Ion Image (Slower)', 'value': 'ion'}
                    ],
                    value=['click'],
                    style={'marginBottom': '15px', 'fontWeight': 'bold', 'display': 'flex',
                           'flexDirection': 'column', 'gap': '5px'}
                ),

                html.Div([
                    html.Label("Target m/z for Ion Image: ", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='mz-dropdown', options=mz_options,
                                 value=sorted_mz_cols[0] if sorted_mz_cols else None, clearable=False,
                                 style={'width': '250px'})
                ], style={'marginBottom': '15px', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),

                html.Div([
                    html.Div(
                        [html.P("Optical Crop", style={'textAlign': 'center', 'margin': '0', 'fontWeight': 'bold'}),
                         html.Img(id="cell-image",
                                  style={"height": "160px", "width": "160px", "backgroundColor": "black",
                                         "borderRadius": "5px"})], style={"marginRight": "10px"}),
                    html.Div(
                        [html.P("Cell Mask", style={'textAlign': 'center', 'margin': '0', 'fontWeight': 'bold'}),
                         html.Img(id="mask-overlay-image",
                                  style={"height": "160px", "width": "160px", "backgroundColor": "black",
                                         "borderRadius": "5px"})], style={"marginRight": "10px"}),
                    html.Div(
                        [html.P("m/z Image", style={'textAlign': 'center', 'margin': '0', 'fontWeight': 'bold'}),
                         html.Img(id="ion-image",
                                  style={"height": "160px", "width": "160px", "backgroundColor": "black",
                                         "borderRadius": "5px"})]),
                ], style={"display": "flex", "flexDirection": "row", "justifyContent": "center"}),

                dcc.Graph(id="spectra-plot", style={"height": "400px", "width": "680px", "marginTop": "20px"})
            ], style={"width": "45%", "padding": "20px", "display": "flex", "flexDirection": "column",
                      "alignItems": "center"}),
        ], style={"display": "flex", "flexDirection": "row"}),
    ])

    # Callback to toggle visibility of standard dropdown vs calculator dropdowns
    @app.callback(
        [Output('standard-color-div', 'style'), Output('ratio-color-div', 'style')],
        [Input('color-mode-toggle', 'value')]
    )
    def toggle_color_mode(mode):
        if mode == 'ratio':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'block'}, {'display': 'none'}

    # FIX: this used to be ONE callback that returned the figure AND the inspection
    # panel, with clickData/hoverData among its inputs. Every hover therefore rebuilt
    # the scatter from scratch, and a rebuilt figure arrives with every trace visible
    # and the axes reset - so any legend entries you had toggled off, and any zoom,
    # snapped back to the default the moment the mouse moved over a point. The figure
    # now has its own callback that only fires when a control that changes the figure
    # changes, and the inspection panel is a separate callback below.
    @app.callback(
        Output("projection-plot", "figure"),
        [Input("projection-dropdown", "value"),
         Input("color-mode-toggle", "value"),
         Input("color-dropdown", "value"),
         Input("ratio-num", "value"),
         Input("ratio-den", "value"),
         Input("ratio-den2", "value"),
         Input("ratio-den3", "value")]
    )
    def update_projection_figure(projection_type, color_mode, color_choice, ratio_num, ratio_den, ratio_den2,
                                 ratio_den3):
        x_col, y_col = f"{projection_type}1", f"{projection_type}2"

        # Denominators 2 and 3 are optional (clearable dropdowns), so drop blanks and any
        # duplicate of an earlier pick so a mass is never counted twice in the total.
        ratio_dens = []
        for d in (ratio_den, ratio_den2, ratio_den3):
            # v5: never the numerator itself (that made every cell exactly 0.5)
            if d and d in df.columns and d not in ratio_dens and d != ratio_num:
                ratio_dens.append(d)

        # FIX: this was df.copy() - a full deep copy of every cell x every column, on
        # EVERY hover event as well as every click. With hover mode on and a large cell
        # count the dashboard becomes unusable. Copy only the columns the figure needs.
        needed = [x_col, y_col, "sample name"]
        if color_mode == 'ratio':
            needed += [ratio_num] + ratio_dens
        else:
            needed += [color_choice]
        needed = [c for c in dict.fromkeys(needed) if c in df.columns]
        df_temp = df.loc[:, needed].copy()

        # FIX: Plotly Express splits a categorical colouring into one trace per category,
        # so clickData's "pointIndex" is an index WITHIN that trace, not into the
        # dataframe. Carry the true positional row index through custom_data instead.
        df_temp["_row_idx"] = np.arange(len(df_temp))

        # --- RATIO CALCULATOR LOGIC ---
        if color_mode == 'ratio':
            # ratio = num / (num + den1 + den2 + den3), bounded 0..1.
            # A zero in ONE denominator must not blank the whole cell (the old
            # replace(0, nan) on both terms would do exactly that once there are three
            # of them), so zeros are kept in the sum and only a zero TOTAL - nothing
            # detected in any of the chosen masses - becomes NaN and is left uncoloured.
            num_vals = pd.to_numeric(df_temp[ratio_num], errors='coerce').fillna(0.0)
            den_total = pd.Series(0.0, index=df_temp.index)
            for d in ratio_dens:
                den_total = den_total + pd.to_numeric(df_temp[d], errors='coerce').fillna(0.0)
            total = (num_vals + den_total).replace(0, np.nan)

            num_str = str(ratio_num).replace('mz_', '')
            den_strs = [str(d).replace('mz_', '') for d in ratio_dens]
            ratio_col_name = f"Ratio: {num_str} / ({' + '.join([num_str] + den_strs)})"

            # notes: Pandas gracefully handles dividing by NaN, outputting a clean NaN value
            df_temp[ratio_col_name] = num_vals / total

            scatter_fig = px.scatter(
                df_temp, x=x_col, y=y_col, color=ratio_col_name,
                color_continuous_scale="Viridis",
                range_color=[0, 1],
                custom_data=["_row_idx"],
                hover_name="sample name",
                template="plotly_white",
                title=f"Custom m/z Ratio Calculator"
            )

        # --- STANDARD LOGIC ---
        else:
            # v3: area_pixels is numeric too, so it gets a continuous colour scale
            if color_choice in mz_cols or color_choice == 'area_pixels':
                df_temp[color_choice] = df_temp[color_choice].astype(float)
                scatter_fig = px.scatter(
                    df_temp, x=x_col, y=y_col, color=color_choice,
                    color_continuous_scale="Viridis",
                    custom_data=["_row_idx"],
                    hover_name="sample name",
                    template="plotly_white"
                )
            else:
                df_temp[color_choice] = df_temp[color_choice].astype(str)
                scatter_fig = px.scatter(
                    df_temp, x=x_col, y=y_col, color=color_choice,
                    # v5: a fixed colour per group, and enough of them for >10 clusters
                    color_discrete_map=(color_map if color_choice in ("Individual Sample Type", "Sample Type + Stim", "Stim")
                                        else group_color_map(df_temp[color_choice].unique())),
                    category_orders={color_choice: (legend_order if legend_order and color_choice == "Individual Sample Type"
                                                    else sorted(df_temp[color_choice].unique(), key=group_sort_key))},
                    custom_data=["_row_idx"],
                    hover_name="sample name",
                    template="plotly_white"
                )

        scatter_fig.update_layout(clickmode='event+select', legend_itemclick="toggle",
                                  legend_itemdoubleclick="toggleothers")
        scatter_fig.update_traces(marker=dict(size=8))

        # tree underneath the cells, aspect squared so drawn distances stay faithful
        # to the tree metric rather than being stretched by the axis ranges
        if projection_type == 'Bonsai' and bonsai_edge_trace is not None:
            scatter_fig.add_trace(bonsai_edge_trace)
            scatter_fig.data = (scatter_fig.data[-1],) + tuple(scatter_fig.data[:-1])
            scatter_fig.update_layout(xaxis=dict(visible=False),
                                      yaxis=dict(visible=False, scaleanchor='x'))

        # Keep zoom/pan and legend toggles when the figure IS legitimately rebuilt
        # (e.g. a colour change). Keyed on the projection so switching to a different
        # embedding, whose axes mean something else, still resets the view.
        scatter_fig.update_layout(uirevision=f"proj-{projection_type}")
        # v5: legend show/hide is remembered per COLOURING - plotly matches traces by
        # position, so a group hidden under one colouring hid an unrelated one under the next
        scatter_fig.update_layout(legend=dict(
            uirevision=f"legend-{projection_type}-{color_mode}-{color_choice}-{ratio_num}"))
        return scatter_fig

    @app.callback(
        [Output("cell-inspection-title", "children"),
         Output("cell-image", "src"),
         Output("mask-overlay-image", "src"),
         Output("ion-image", "src"),
         Output("spectra-plot", "figure")],
        [Input("projection-plot", "clickData"),
         Input("projection-plot", "hoverData"),
         Input("mz-dropdown", "value"),
         Input("interaction-checkboxes", "value")]
    )
    def update_inspection(click_data, hover_data, mz_value, interaction_modes):
        modes = interaction_modes or []
        # v5: with the selection locked to clicks, a hover changes nothing - stop it before
        # it reloads the masks and the h5 for the cell that is already shown
        _triggered = [t_['prop_id'] for t_ in (callback_context.triggered or [])]
        if 'click' in modes and _triggered and all(p_.endswith('.hoverData') for p_ in _triggered):
            raise PreventUpdate
        active_data = click_data if 'click' in modes else hover_data

        if not active_data:
            return "Cell Inspection (Hover or Click over a cell)", None, None, None, go.Figure()

        # FIX: prefer the custom_data row index; fall back to pointIndex only if a trace
        # somehow arrives without it (which is correct for single-trace figures anyway).
        _point = active_data["points"][0]
        _custom = _point.get("customdata")
        if _custom:
            idx = int(_custom[0])
        else:
            idx = int(_point["pointIndex"])
        row = df.iloc[idx]
        sample_name = str(row["sample name"])
        mask_id = int(row["mask_id"])

        dynamic_title = f"Cell Inspection: {sample_name} (Cell {mask_id})"
        if "area_pixels" in row.index:
            try:
                dynamic_title += f" - {float(row['area_pixels']):.0f} px"
            except (TypeError, ValueError):
                pass

        raw_src = None
        overlay_src = None
        ion_src = None

        h5_path = Path(master_dir) / sample_name / "PREMSI_stage1_ready.h5"
        seg_path = Path(master_dir) / sample_name / "PREMSI_seg.npy"

        if h5_path.exists() and seg_path.exists():
            try:
                masks = _load_masks_cached(str(seg_path), seg_path.stat().st_mtime)

                with h5py.File(str(h5_path), 'r') as f:
                    # Stage 2 now writes its rotation/translation correction to a separate
                    # dataset rather than overwriting stage 1's. Prefer it when present.
                    imgs_cp = f["imgs_cp_aligned"] if "imgs_cp_aligned" in f else f["imgs_cp"]
                    raw_cube = f["raw_cube"]
                    masses = f["masses_of_interest"][:]

                    CROP_SIZE = 100
                    cy, cx = int(row["centroid_row"]), int(row["centroid_col"])
                    y1, y2 = max(0, cy - CROP_SIZE), min(imgs_cp.shape[0], cy + CROP_SIZE)
                    x1, x2 = max(0, cx - CROP_SIZE), min(imgs_cp.shape[1], cx + CROP_SIZE)

                    img_crop = imgs_cp[y1:y2, x1:x2].astype(np.float32)

                    if img_crop.ndim == 3 and img_crop.shape[0] <= 5:
                        img_crop = np.moveaxis(img_crop, 0, -1)

                    max_val = np.max(img_crop)
                    if max_val > 0: img_crop /= max_val

                    CHANNEL_COLORS = {0: (1, 1, 1), 1: (1, 0, 0), 2: (0, 0, 1)}
                    final_composite = np.zeros((img_crop.shape[0], img_crop.shape[1], 3), dtype=np.float32)
                    n_channels = img_crop.shape[2] if img_crop.ndim == 3 else 1

                    if img_crop.ndim == 2: img_crop = np.expand_dims(img_crop, axis=-1)

                    for ch_idx, color_tuple in CHANNEL_COLORS.items():
                        if 0 <= ch_idx < n_channels:
                            final_composite += img_crop[:, :, ch_idx, np.newaxis] * np.array(color_tuple)

                    crop_bgr = (np.clip(final_composite[:, :, ::-1], 0, 1) * 255).astype(np.uint8)

                    _, buffer_raw = cv2.imencode('.png', crop_bgr)
                    raw_src = f"data:image/png;base64,{base64.b64encode(buffer_raw.tobytes()).decode()}"

                    mask_crop = masks[y1:y2, x1:x2]
                    overlay = crop_bgr.copy()
                    local_cy, local_cx = cy - y1, cx - x1

                    # FIX: this used to re-derive the cell ID from whichever pixel sat under
                    # the centroid. For a concave or branched cell (i.e. every neuron) the
                    # centroid can land in background or inside a NEIGHBOURING cell, so the
                    # wrong cell got highlighted. mask_id from the row is the true Cellpose
                    # ID, so use it directly and only fall back to the pixel lookup if that
                    # ID isn't present in this crop.
                    target_mask_id = mask_id if np.any(mask_crop == mask_id) else 0
                    if target_mask_id == 0 and 0 <= local_cy < mask_crop.shape[0] and 0 <= local_cx < mask_crop.shape[1]:
                        target_mask_id = int(mask_crop[local_cy, local_cx])

                    if target_mask_id > 0:
                        highlight_mask = (mask_crop == target_mask_id).astype(np.uint8)
                        color_mask = np.zeros_like(overlay)
                        cluster_val = row["Individual Sample Type"]

                        cmap = color_map if color_map else {}
                        c_str = cmap.get(cluster_val, "#ffffff")
                        h_hex = c_str.lstrip('#')
                        cluster_bgr = tuple(int(h_hex[i:i + 2], 16) for i in (0, 2, 4))[::-1]

                        if highlight_mask.shape[:2] == color_mask.shape[:2]:
                            color_mask[highlight_mask > 0] = cluster_bgr
                            cv2.addWeighted(color_mask, 0.4, overlay, 0.6, 0, overlay)
                            cnts, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, cnts, -1, cluster_bgr, 2)

                    _, buffer_overlay = cv2.imencode('.png', overlay)
                    overlay_src = f"data:image/png;base64,{base64.b64encode(buffer_overlay.tobytes()).decode()}"

                    if 'ion' in modes and mz_value:
                        try:
                            target_mz = float(str(mz_value).replace('mz_', ''))
                            idx_mass = int((np.abs(masses - target_mz)).argmin())
                            ion_crop = raw_cube[y1:y2, x1:x2, idx_mass]

                            ion_crop = np.squeeze(ion_crop).astype(np.float32)

                            if ion_crop.max() > 0:
                                ion_norm = (ion_crop / ion_crop.max() * 255).astype(np.uint8)
                            else:
                                ion_norm = np.zeros_like(ion_crop, dtype=np.uint8)

                            ion_color = cv2.applyColorMap(ion_norm, cv2.COLORMAP_INFERNO)
                            _, buffer_ion = cv2.imencode('.png', ion_color)
                            ion_src = f"data:image/png;base64,{base64.b64encode(buffer_ion.tobytes()).decode()}"
                        except Exception as e:
                            print(f"Error generating ion image: {e}")
                            ion_src = None
            except Exception as e:
                # Image rendering must never take the spectrum down with it. Before this
                # guard any failure in the crop/mask/h5 block aborted the whole callback,
                # so all six outputs (including spectra-plot) silently stopped updating.
                print(f'[DASH] Cell image failed for {sample_name} (cell {mask_id}): {e}')
                raw_src, overlay_src, ion_src = None, None, None


        # A category axis with one tick per mass is unusable (and slow enough to look
        # broken) once the list is long, so the m/z axis is numeric and the bars are
        # width-scaled. Zero features are dropped rather than drawn as a solid floor.
        try:
            mz_values = np.array([float(str(c).replace('mz_', '')) for c in sorted_mz_cols])
            intensities = pd.to_numeric(
                pd.Series([row[c] if c in row.index else 0.0 for c in sorted_mz_cols]),
                errors='coerce').fillna(0.0).values
            nonzero = intensities != 0
            if nonzero.any():
                mz_plot, int_plot = mz_values[nonzero], intensities[nonzero]
            else:
                mz_plot, int_plot = mz_values, intensities

            spec_fig = go.Figure(data=go.Bar(
                x=mz_plot, y=int_plot, marker_color='royalblue',
                width=max(0.05, float(np.ptp(mz_values)) / max(len(mz_values), 1) * 0.8),
                hovertemplate='m/z %{x:.4f}<br>%{y:.4g}<extra></extra>'))
            spec_fig.update_layout(
                title=f'm/z Spectrum: {sample_name} (Cell {mask_id}) - '
                      f'{int(nonzero.sum())}/{len(mz_values)} non-zero',
                margin=dict(l=20, r=20, t=60, b=40),
                xaxis_title='m/z', yaxis_title='TIC Normalized Intensity',
                template='plotly_white', bargap=0, xaxis=dict(type='linear'))
        except Exception as e:
            print(f'[DASH] Spectrum failed for {sample_name} (cell {mask_id}): {e}')
            spec_fig = go.Figure()
            spec_fig.update_layout(title=f'Spectrum unavailable: {e}',
                                   template='plotly_white')

        return dynamic_title, raw_src, overlay_src, ion_src, spec_fig

    safe_port = find_free_port(port)
    Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{safe_port}")).start()
    app.run(port=safe_port, debug=False)
