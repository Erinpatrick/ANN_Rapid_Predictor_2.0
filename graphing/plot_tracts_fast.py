"""
Fast 3D visualization for large tractography datasets.

Reads ANN prediction results and the corresponding tract file, then generates
interactive Plotly HTML plots showing activated vs. inactive fibers with an
optional electric-field isosurface from the electrode FEM data.

Key design: Only TWO Plotly traces are created for fibers (one red "Activated",
one grey "Inactive"), with individual fibers merged into single coordinate arrays
separated by NaN.  This lets the browser render 100k+ fibers in real-time.
"""

import os
import sys
import json
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def mkdirp(path):
    os.makedirs(path, exist_ok=True)


def read_tract_file(path, downsample=1):
    """Read a tract text file.  Each line = one fiber: x1 y1 z1 x2 y2 z2 ...
    Returns a list of numpy arrays, each with shape (N, 3)."""
    print(f"Reading tract file: {path} ...")
    t0 = time.time()
    fibers = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = np.fromstring(line, sep=" ")
            if nums.size % 3 != 0:
                continue
            coords = nums.reshape(-1, 3)
            if downsample > 1:
                coords = coords[::downsample]
            fibers.append(coords)
    print(f"Loaded {len(fibers)} fibers in {time.time() - t0:.2f}s")
    return fibers


def load_results(path):
    """Load a results JSON written by dti_ann_LUT.py.

    Returns
    -------
    data : dict          - the raw JSON
    pulse_widths : list   - sorted list of pulse-width keys found (as floats, in seconds)
    valid_inds : list     - list of valid fiber indices (may be empty)
    """
    with open(path) as f:
        data = json.load(f)

    valid_inds = data.get("valid_inds", [])

    # Discover pulse widths present in the file.  They are stored as string
    # keys like "0.06" (seconds).  Skip non-numeric keys.
    pw_keys = []
    for k in data:
        try:
            pw_keys.append(float(k))
        except ValueError:
            pass
    pw_keys.sort()

    return data, pw_keys, [int(i) for i in valid_inds]


def get_thresholds_for_pw(data, pw_key):
    """Return a list of threshold values for a given pulse-width key."""
    pw_data = data.get(str(pw_key), data.get(pw_key, {}))
    if isinstance(pw_data, list):
        return pw_data
    if isinstance(pw_data, dict):
        indices = sorted(int(k) for k in pw_data)
        return [pw_data[str(i)] for i in indices]
    return []


# ---------------------------------------------------------------------------
# Electric field loading (subsampled for visualisation)
# ---------------------------------------------------------------------------

def load_electrode_field(path, subsample=4, electrode_center=None):
    """Load a COMSOL electrode export and return subsampled 3D field data.

    Parameters
    ----------
    path : str
        Path to the electrode .txt file (COMSOL export: x y z V).
    subsample : int
        Keep every Nth unique coordinate along each axis to reduce data volume.
    electrode_center : tuple or None
        (cx, cy, cz) — if provided, the FEM grid is re-centered so its
        midpoint sits at these coordinates (same logic as dti_ann_LUT.py).

    Returns
    -------
    X, Y, Z : 3-D ndarrays (meshgrid)
    V : 3-D ndarray of electric potential (mV, absolute value, log-scaled)
    """
    print(f"Loading electrode field: {path} ...")
    t0 = time.time()

    x_coords, y_coords, z_coords, potentials = [], [], [], []
    x_prev = y_prev = z_prev = None

    with open(path) as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            xv = round(float(parts[0]), 3)
            yv = round(float(parts[1]), 3)
            zv = round(float(parts[2]), 3)
            vv = float(parts[3])
            potentials.append(vv)
            if xv != x_prev and xv not in x_coords:
                x_coords.append(xv)
            if yv != y_prev and yv not in y_coords:
                y_coords.append(yv)
            if zv != z_prev and zv not in z_coords:
                z_coords.append(zv)
            x_prev, y_prev, z_prev = xv, yv, zv

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    print(f"  Grid: {nx} x {ny} x {nz} = {nx*ny*nz:,} points")

    # Reshape — data is stored z-outer, y-middle, x-inner (like FEM.py)
    V_full = np.array(potentials, dtype=np.float64).reshape((nz, ny, nx))
    V_full = np.transpose(V_full, (2, 1, 0))  # -> (nx, ny, nz)

    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)
    z_arr = np.array(z_coords)

    # Subsample
    xs = x_arr[::subsample]
    ys = y_arr[::subsample]
    zs = z_arr[::subsample]
    Vs = V_full[::subsample, ::subsample, ::subsample]

    # Apply centering offset (same as dti_ann_LUT.py)
    if electrode_center is not None:
        orig_cx = (x_arr[0] + x_arr[-1]) / 2.0
        orig_cy = (y_arr[0] + y_arr[-1]) / 2.0
        orig_cz = (z_arr[0] + z_arr[-1]) / 2.0
        xs = xs + (electrode_center[0] - orig_cx)
        ys = ys + (electrode_center[1] - orig_cy)
        zs = zs + (electrode_center[2] - orig_cz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    print(f"  Subsampled to {len(xs)} x {len(ys)} x {len(zs)} = {Vs.size:,} points "
          f"in {time.time() - t0:.1f}s")

    return X, Y, Z, Vs


# ---------------------------------------------------------------------------
# Batched trace building (the performance trick)
# ---------------------------------------------------------------------------

def build_merged_trace(fibers, indices, pts_stride=1):
    """Merge many fibers into one (x, y, z) tuple with NaN separators.

    Parameters
    ----------
    pts_stride : int
        Keep every Nth point per fiber (1 = full resolution).
    """
    if not indices:
        return None, None, None
    selected = [fibers[i][::pts_stride] for i in indices]
    total = sum(f.shape[0] for f in selected) + len(selected)
    merged = np.full((total, 3), np.nan, dtype=np.float32)
    cur = 0
    for fib in selected:
        n = fib.shape[0]
        merged[cur : cur + n, :] = fib
        cur += n + 1
    return merged[:, 0], merged[:, 1], merged[:, 2]


# ---------------------------------------------------------------------------
# Electrode config parsing
# ---------------------------------------------------------------------------

def parse_electrode_config(config_str):
    """Parse an electrode configuration string like '01-23', '+012-3', '-0+1-2+3'.

    Returns a dict mapping contact number (0-3) to polarity:
        '+' = anode (blue), '-' = cathode (red), None = inactive (grey).

    Rules:
        - A '+' or '-' sign applies to the NEXT digit only.
        - Digits without a preceding sign are inactive.
    """
    contact_polarity = {0: None, 1: None, 2: None, 3: None}
    current_sign = None
    for ch in config_str:
        if ch in '+-':
            current_sign = ch
        elif ch.isdigit():
            contact_polarity[int(ch)] = current_sign
            current_sign = None  # sign consumed — reset
    return contact_polarity


def contact_color(polarity):
    """Return a display color for a contact given its polarity."""
    if polarity == '-':
        return '#CC0000'   # red  (cathode)
    elif polarity == '+':
        return '#0066CC'   # blue (anode)
    return '#404040'       # grey (inactive)


def contact_label(index, polarity):
    if polarity == '-':
        return f'Contact {index} (cathode -)'
    elif polarity == '+':
        return f'Contact {index} (anode +)'
    return f'Contact {index} (inactive)'


# ---------------------------------------------------------------------------
# Electric field rendering
# ---------------------------------------------------------------------------

def add_electric_field(fig, X, Y, Z, V):
    """Add electric-field isosurface rendering to the figure.

    Creates compact sphere-like isosurface shells around each electrode
    contact where the field is strongest.  Automatically adapts to any
    electrode configuration (monopolar, bipolar, tripolar, quadrupolar)
    by rendering separate cathode (red) and anode (blue) isosurfaces
    around the peak field values.

    Returns the number of Plotly traces added (needed for toggle button).
    """
    Vc = V.copy()
    Vc[np.isnan(Vc)] = 0.0

    vmin = float(np.min(Vc))
    vmax = float(np.max(Vc))
    abs_peak = max(abs(vmin), abs(vmax))

    if abs_peak < 1e-12:
        return 0

    only_negative = vmax <= 1e-10
    only_positive = vmin >= -1e-10

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    v_flat = Vc.flatten()

    # Fraction of each pole's peak value used as the isosurface range.
    # e.g. 0.50 means show the region where |V| > 50% of |V_peak|.
    peak_frac = 0.50
    n_iso = 3          # nested isosurface shells per pole

    traces_added = 0

    # ── Unified colorbar via an invisible marker ────────────────────
    if only_negative:
        cb_cmin, cb_cmax = vmin, 0
        cb_cscale = [
            [0.0, 'rgb(103,0,31)'],   [0.25, 'rgb(178,24,43)'],
            [0.5, 'rgb(214,96,77)'],  [0.75, 'rgb(244,165,130)'],
            [1.0, 'rgb(253,219,199)'],
        ]
        cb_tickvals = [vmin, 0]
        cb_ticktext = ["-max", "0"]
    elif only_positive:
        cb_cmin, cb_cmax = 0, vmax
        cb_cscale = [
            [0.0, 'rgb(209,229,240)'], [0.25, 'rgb(146,197,222)'],
            [0.5, 'rgb(67,147,195)'],  [0.75, 'rgb(33,102,172)'],
            [1.0, 'rgb(5,48,97)'],
        ]
        cb_tickvals = [0, vmax]
        cb_ticktext = ["0", "+max"]
    else:
        cb_cmin, cb_cmax = -abs_peak, abs_peak
        cb_cscale = 'RdBu'
        cb_tickvals = [-abs_peak, 0, abs_peak]
        cb_ticktext = ["-max", "0", "+max"]

    cx = float(np.mean(x_flat))
    cy = float(np.mean(y_flat))
    cz = float(np.mean(z_flat))
    fig.add_trace(go.Scatter3d(
        x=[cx], y=[cy], z=[cz],
        mode='markers',
        marker=dict(
            size=0.001, opacity=0,
            color=[0],
            colorscale=cb_cscale,
            cmin=cb_cmin, cmax=cb_cmax,
            showscale=True,
            colorbar=dict(title="E-field", x=0.02, len=0.6,
                         tickvals=cb_tickvals, ticktext=cb_ticktext),
        ),
        showlegend=False, hoverinfo='skip', legendgroup="efield",
    ))
    traces_added += 1

    hover_tpl = (
        "x: %{x:.1f}<br>"
        "y: %{y:.1f}<br>"
        "z: %{z:.1f}<br>"
        "V: %{value:.4g} V"
    )

    # ── Cathode (negative) isosurface ───────────────────────────────
    if vmin < -1e-10:
        neg_peak = abs(vmin)
        iso_max_neg = -(neg_peak * (1.0 - peak_frac))
        red_cs = [
            [0.0, 'rgb(103,0,31)'],
            [0.5, 'rgb(178,24,43)'],
            [1.0, 'rgb(244,165,130)'],
        ]
        fig.add_trace(go.Isosurface(
            x=x_flat, y=y_flat, z=z_flat, value=v_flat,
            isomin=vmin, isomax=iso_max_neg,
            surface_count=n_iso,
            colorscale=red_cs,
            opacity=0.6,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
            name="Electric Field",
            visible=True, legendgroup="efield", showlegend=True,
            hovertemplate=hover_tpl + "<extra>E-field (cathode)</extra>",
        ))
        traces_added += 1

    # ── Anode (positive) isosurface ─────────────────────────────────
    if vmax > 1e-10:
        pos_peak = abs(vmax)
        iso_min_pos = pos_peak * (1.0 - peak_frac)
        blue_cs = [
            [0.0, 'rgb(209,229,240)'],
            [0.5, 'rgb(67,147,195)'],
            [1.0, 'rgb(5,48,97)'],
        ]
        fig.add_trace(go.Isosurface(
            x=x_flat, y=y_flat, z=z_flat, value=v_flat,
            isomin=iso_min_pos, isomax=vmax,
            surface_count=n_iso,
            colorscale=blue_cs,
            opacity=0.6,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
            name="Electric Field",
            visible=True, legendgroup="efield", showlegend=False,
            hovertemplate=hover_tpl + "<extra>E-field (anode)</extra>",
        ))
        traces_added += 1

    return traces_added


# ---------------------------------------------------------------------------
# Interactive HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE_FAST = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fiber Activation Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
#controls {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 10px 16px; background: #f5f5f5; border-bottom: 1px solid #ddd;
}
#controls label { font-size: 14px; font-weight: 600; color: #333; }
#controls select, #controls input[type=number] {
  padding: 5px 10px; font-size: 14px; border: 1px solid #aaa; border-radius: 4px;
}
#controls input[type=number] { width: 90px; }
#controls button {
  padding: 6px 18px; font-size: 14px; background: #4CAF50; color: white;
  border: none; border-radius: 4px; cursor: pointer;
}
#controls button:hover { background: #45a049; }
#status { margin-left: auto; font-size: 13px; color: #666; }
#plot { width: 100%; height: calc(100vh - 52px); }
</style>
</head>
<body>
<div id="controls">
  <label>Pulse Width:
    <select id="pw-select" onchange="buildPlot()">
___PW_OPTIONS___
    </select>
  </label>
  <label>Voltage (V):
    <input id="voltage-input" type="number" value="___DEFAULT_VOLTAGE___" step="0.1" min="0" oninput="syncMA(this)">
  </label>
  <label>Current (mA):
    <input id="ma-input" type="number" value="___DEFAULT_MA___" step="0.1" min="0" oninput="syncV(this)">
  </label>
  <button onclick="buildPlot()">Apply</button>
  <span id="status"></span>
</div>
<div id="plot"></div>
<script>
var FIBERS = ___FIBERS_DATA___;
var THRESHOLDS = ___THRESHOLDS_DATA___;
var EFIELD_TRACES = ___EFIELD_DATA___;
var SHOW_AXES = ___SHOW_AXES___;
var V_TO_MA = 1.123;

function syncMA(vInput) {
  var v = parseFloat(vInput.value);
  if (!isNaN(v)) document.getElementById("ma-input").value = (v * V_TO_MA).toFixed(3);
}
function syncV(maInput) {
  var ma = parseFloat(maInput.value);
  if (!isNaN(ma)) document.getElementById("voltage-input").value = (ma / V_TO_MA).toFixed(4);
}

function buildMergedTrace(indices, color, width, opacity, name, visible, legendgroup) {
  var x = [], y = [], z = [];
  for (var ii = 0; ii < indices.length; ii++) {
    var fib = FIBERS[indices[ii]];
    for (var j = 0; j < fib.length; j += 3) {
      x.push(fib[j]); y.push(fib[j+1]); z.push(fib[j+2]);
    }
    x.push(null); y.push(null); z.push(null);
  }
  return {
    type: "scatter3d", mode: "lines",
    x: x, y: y, z: z,
    line: {color: color, width: width},
    opacity: opacity, name: name,
    connectgaps: false, showlegend: true,
    visible: visible, legendgroup: legendgroup
  };
}

function buildPlot() {
  var pwKey = document.getElementById("pw-select").value;
  var voltage = parseFloat(document.getElementById("voltage-input").value);
  if (isNaN(voltage) || voltage <= 0) return;
  var gd = document.getElementById("plot");
  var savedCamera = null;
  var savedVis = {};
  if (gd.data && gd.data.length > 0) {
    for (var t = 0; t < gd.data.length; t++) {
      var lg = gd.data[t].legendgroup;
      if (lg) savedVis[lg] = gd.data[t].visible;
    }
  }
  if (gd.layout && gd.layout.scene && gd.layout.scene.camera) {
    savedCamera = JSON.parse(JSON.stringify(gd.layout.scene.camera));
  }
  document.getElementById("status").textContent = "Building plot...";
  setTimeout(function() {
    var thresholds = THRESHOLDS[pwKey];
    var activeIdxs = [], inactiveIdxs = [];
    for (var i = 0; i < FIBERS.length; i++) {
      var thr = (i < thresholds.length && thresholds[i] !== null) ? thresholds[i] : Infinity;
      if (thr < voltage) activeIdxs.push(i); else inactiveIdxs.push(i);
    }
    var traces = [];
    if (activeIdxs.length > 0)
      traces.push(buildMergedTrace(activeIdxs, "red", 2, 1.0, "Activated", true, "activated"));
    if (inactiveIdxs.length > 0)
      traces.push(buildMergedTrace(inactiveIdxs, "black", 1, 0.15, "Inactive", "legendonly", "inactive"));
    var efStart = traces.length;
    for (var k = 0; k < EFIELD_TRACES.length; k++) {
      var tc = {};
      for (var key in EFIELD_TRACES[k]) tc[key] = EFIELD_TRACES[k][key];
      traces.push(tc);
    }
    var axStyle = SHOW_AXES
      ? {visible:true, showticklabels:true, showgrid:true, zeroline:true, title:""}
      : {visible:false, showticklabels:false, showgrid:false, zeroline:false};
    var layout = {
      title: {text: "Pulse Width: " + pwKey + " \u03bcs  |  Threshold: " + voltage.toFixed(4) + " V  /  " + (voltage * V_TO_MA).toFixed(3) + " mA", x: 0.5, xanchor: "center"},
      scene: {
        camera: {eye: {x:1.5, y:1.5, z:1.5}},
        aspectmode: "data",
        xaxis: axStyle, yaxis: axStyle, zaxis: axStyle
      },
      showlegend: true,
      legend: {itemsizing: "constant", title: {text: "Click to toggle"}},
      margin: {l:60, r:0, t:30, b:0}
    };
    if (EFIELD_TRACES.length > 0) {
      var efIdxs = [];
      for (var ei = efStart; ei < traces.length; ei++) efIdxs.push(ei);
      layout.updatemenus = [{
        type: "buttons",
        buttons: [{
          label: "Toggle Electric Field",
          method: "restyle",
          args: [{visible: efIdxs.map(function(){return false;})}, efIdxs],
          args2: [{visible: efIdxs.map(function(){return true;})}, efIdxs]
        }],
        showactive: true, x: 0.0, xanchor: "left", y: 1.05, yanchor: "top"
      }];
    }
    for (var t = 0; t < traces.length; t++) {
      var lg = traces[t].legendgroup;
      if (lg && savedVis[lg] !== undefined) traces[t].visible = savedVis[lg];
    }
    if (savedCamera) layout.scene.camera = savedCamera;
    Plotly.react("plot", traces, layout);
    document.getElementById("status").textContent =
      "Activated: " + activeIdxs.length + " / " + FIBERS.length + " fibers";
  }, 50);
}

document.getElementById("voltage-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") buildPlot();
});
document.getElementById("ma-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") buildPlot();
});
buildPlot();
</script>
</body>
</html>"""  # end HTML_TEMPLATE_FAST


def prepare_efield_traces(field_data):
    """Build E-field Plotly traces and return as a JSON string."""
    if field_data is None:
        return "[]"
    import plotly.io as pio
    X, Y, Z, V = field_data
    temp_fig = go.Figure()
    add_electric_field(temp_fig, X, Y, Z, V)
    fig_dict = json.loads(pio.to_json(temp_fig))
    return json.dumps(fig_dict['data'])


def write_interactive_html(path, fiber_coords, all_thresholds, pw_options,
                            efield_traces_json, show_axes, default_voltage=5.0):
    """Write a single interactive HTML viewer with PW dropdown and voltage input."""
    opts_html = "\n".join(
        f'      <option value="{key}">{label}</option>' for key, label in pw_options
    )
    html = HTML_TEMPLATE_FAST
    html = html.replace("___PW_OPTIONS___", opts_html)
    html = html.replace("___FIBERS_DATA___", json.dumps(fiber_coords))
    html = html.replace("___THRESHOLDS_DATA___", json.dumps(all_thresholds))
    html = html.replace("___EFIELD_DATA___", efield_traces_json)
    html = html.replace("___SHOW_AXES___", "true" if show_axes else "false")
    html = html.replace("___DEFAULT_VOLTAGE___", str(default_voltage))
    html = html.replace("___DEFAULT_MA___", str(round(default_voltage * 1.123, 3)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote interactive HTML: {path}")


# ---------------------------------------------------------------------------
# Scene rendering (kept for backward compatibility)
# ---------------------------------------------------------------------------

def render(fibers, thresholds, voltage, electrode_center, title="",
           show_axes=False, electrode_config=None, field_data=None,
           max_simplify=3):
    """Build a Plotly figure with activated/inactive fibers and optional E-field."""
    fig = go.Figure()

    active, inactive = [], []
    for i in range(len(fibers)):
        thr = float("inf")
        if i < len(thresholds):
            try:
                thr = float(thresholds[i])
            except (TypeError, ValueError):
                pass
        (active if thr < voltage else inactive).append(i)

    print(f"  Activated: {len(active)},  Inactive: {len(inactive)}")

    stride = max(1, max_simplify)
    if stride > 1:
        print(f"  Simplifying fibers: keeping every {stride}th point")

    ax, ay, az = build_merged_trace(fibers, active, pts_stride=stride)
    if ax is not None:
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az, mode="lines",
            line=dict(color="red", width=2), opacity=1.0,
            name="Activated", connectgaps=False,
            legendgroup="activated", showlegend=True,
        ))

    ix, iy, iz = build_merged_trace(fibers, inactive, pts_stride=stride)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode="lines",
            line=dict(color="black", width=1), opacity=0.15,
            name="Inactive", connectgaps=False,
            legendgroup="inactive", showlegend=True,
            visible="legendonly",  # inactive fibers hidden by default
        ))

    # Electric field (replaces lead mesh)
    n_before_efield = len(fig.data)
    if field_data is not None:
        X, Y, Z, V = field_data
        add_electric_field(fig, X, Y, Z, V)

    # Axes toggle visibility
    show = show_axes
    axis_style = dict(
        visible=True,
        showticklabels=show,
        showgrid=show,
        zeroline=show,
        title="" if not show else None,
    )
    hide = dict(visible=False, showticklabels=False, showgrid=False, zeroline=False)

    scene = dict(
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        aspectmode="data",
        xaxis=axis_style if show else hide,
        yaxis=axis_style if show else hide,
        zaxis=axis_style if show else hide,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=scene,
        showlegend=True,
        legend=dict(
            itemsizing="constant",
            title="Click to toggle",
        ),
        margin=dict(l=60, r=0, t=30, b=0),
    )

    # Add E-field toggle button
    if field_data is not None:
        efield_indices = list(range(n_before_efield, len(fig.data)))
        n_ef = len(efield_indices)
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(
                    label="Toggle Electric Field",
                    method="restyle",
                    args=[{"visible": [False] * n_ef}, efield_indices],
                    args2=[{"visible": [True] * n_ef}, efield_indices],
                )],
                showactive=True,
                x=0.0, xanchor="left",
                y=1.05, yanchor="top",
            )]
        )

    return fig, len(active)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast 3D fiber-activation visualizer for ANN Rapid Predictor results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Plot all pulse widths found in the results file
  python plot_tracts_fast.py --tract tracts.txt --results results.json --output plots/

  # Only show fibers activated below 2.5 V
  python plot_tracts_fast.py --tract tracts.txt --results results.json --output plots/ --activation_threshold 2.5

  # Downsample points per fiber for faster rendering of huge files
  python plot_tracts_fast.py --tract tracts.txt --results results.json --output plots/ --downsample 3
""",
    )

    parser.add_argument(
        "--tract", required=True, metavar="FILE",
        help="Path to the tract text file (one fiber per line, x1 y1 z1 x2 y2 z2 ...).",
    )
    parser.add_argument(
        "--results", required=True, metavar="FILE",
        help="Path to the JSON results file produced by dti_ann_LUT.py.",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Directory where HTML plots will be saved.",
    )
    parser.add_argument(
        "--activation_threshold", type=float, default=5.0, metavar="VOLTS",
        help="Default voltage (V) below which a fiber is considered activated.  "
             "Can be changed interactively in the HTML viewer.  Default: 5.0",
    )
    parser.add_argument(
        "--electrode_center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="X Y Z position of the electrode center for display.  "
             "If omitted, defaults to (0, 0, 0).",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, metavar="N",
        help="Keep every Nth point per fiber to speed up rendering.  Default: 1 (no downsampling).",
    )
    parser.add_argument(
        "--simplify", type=int, default=3, metavar="N",
        help="Keep every Nth point per fiber to reduce rendering load. "
             "Default: 3. Set to 1 for full resolution.",
    )
    parser.add_argument(
        "--show_axes", action="store_true",
        help="Show X/Y/Z axes in the 3D scene.",
    )
    parser.add_argument(
        "--all_fibers", action="store_true",
        help="Plot ALL fibers at full length, ignoring the FEM valid_inds filter. "
             "Fibers without a threshold are drawn as inactive.",
    )
    parser.add_argument(
        "--electrode_config", type=str, default=None, metavar="CONFIG",
        help="Electrode contact configuration string, e.g. '01-23', '+012-3', '-0+1-2+3'. "
             "'-' marks cathodes (red), '+' marks anodes (blue), unmarked contacts are grey.",
    )
    parser.add_argument(
        "--electrode", type=str, default=None, metavar="FILE",
        help="Path to the electrode FEM export file (.txt, COMSOL format). "
             "When provided, an electric-field isosurface is shown instead of a "
             "lead mesh.  Toggle it on/off in the legend.",
    )
    parser.add_argument(
        "--field_subsample", type=int, default=4, metavar="N",
        help="Subsample the electrode grid by keeping every Nth point per axis. "
             "Default: 4.  Increase to reduce memory / rendering time.",
    )

    args = parser.parse_args()
    mkdirp(args.output)

    electrode_center = tuple(args.electrode_center) if args.electrode_center else (0, 0, 0)
    electrode_config = parse_electrode_config(args.electrode_config) if args.electrode_config else None

    # Load electric field if an electrode file was provided
    field_data = None
    if args.electrode:
        field_data = load_electrode_field(
            args.electrode,
            subsample=args.field_subsample,
            electrode_center=electrode_center,
        )

    # 1. Load results JSON - auto-detect pulse widths
    print(f"Loading results: {args.results}")
    data, pw_keys, valid_inds = load_results(args.results)

    if not pw_keys:
        print("ERROR: No pulse-width data found in the results file.")
        sys.exit(1)

    print(f"Found {len(pw_keys)} pulse width(s): " +
          ", ".join(f"{pw*1000:.0f} us" for pw in pw_keys))

    # 2. Read fibers
    fibers = read_tract_file(args.tract, downsample=args.downsample)

    # 3. Filter to valid indices (if present in results) or show all
    if args.all_fibers:
        print(f"--all_fibers: showing all {len(fibers)} fibers (no FEM filter).")
    elif valid_inds:
        print(f"Filtering to {len(valid_inds)} valid fibers (from results file)...")
        try:
            fibers = [fibers[i] for i in valid_inds]
        except IndexError:
            print("ERROR: valid_inds references fiber indices that don't exist in the tract file.")
            sys.exit(1)
        print(f"  {len(fibers)} fibers after filtering.")

    # 4. Prepare data for interactive HTML
    stride = max(1, args.simplify)
    print(f"\nPreparing fiber data (simplify={stride}) ...")
    fiber_coords = []
    for fib in fibers:
        coords = fib[::stride] if stride > 1 else fib
        fiber_coords.append(np.round(coords, 2).flatten().tolist())

    # Collect thresholds per PW (aligned to display fiber indices)
    all_thresholds = {}
    pw_options = []
    for pw in pw_keys:
        pw_us = f"{pw * 1000:.0f}"
        thr_list = get_thresholds_for_pw(data, pw)
        padded = []
        for i in range(len(fibers)):
            if i < len(thr_list):
                try:
                    padded.append(round(float(thr_list[i]), 4))
                except (TypeError, ValueError):
                    padded.append(None)
            else:
                padded.append(None)
        all_thresholds[pw_us] = padded
        pw_options.append((pw_us, f"{pw_us} \u03bcs"))

    # E-field traces
    efield_json = prepare_efield_traces(field_data)

    # Generate single interactive HTML
    out_html = os.path.join(args.output, "activation.html")
    write_interactive_html(
        out_html, fiber_coords, all_thresholds, pw_options,
        efield_json, args.show_axes, default_voltage=args.activation_threshold,
    )
    print(f"\nDone. Interactive viewer -> {out_html}")


if __name__ == "__main__":
    main()
