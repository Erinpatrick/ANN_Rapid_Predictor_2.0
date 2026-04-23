import os
import sys
import json
import math
import time
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]


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

    V_full = np.array(potentials, dtype=np.float64).reshape((nz, ny, nx))
    V_full = np.transpose(V_full, (2, 1, 0))  # -> (nx, ny, nz)

    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)
    z_arr = np.array(z_coords)

    xs = x_arr[::subsample]
    ys = y_arr[::subsample]
    zs = z_arr[::subsample]
    Vs = V_full[::subsample, ::subsample, ::subsample]

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

HTML_TEMPLATE = r"""<!DOCTYPE html>
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
</html>"""  # end HTML_TEMPLATE


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
    html = HTML_TEMPLATE
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


def parse_electrode_config(config_str):
    """Parse an electrode configuration string like '01-23', '+012-3', '-0+1-2+3'.

    Returns a dict mapping contact number (0-3) to polarity:
        '+' = anode (blue), '-' = cathode (red), None = inactive (grey).
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


def polarity_to_color(polarity, lead_color, inactive_color):
    """Map polarity to display color."""
    if polarity == '-':
        return '#CC0000'   # red  (cathode)
    elif polarity == '+':
        return '#0066CC'   # blue (anode)
    return inactive_color  # grey (inactive)

def mkdirp(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def read_tract_file(path):
    fibers = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            nums = [float(x) for x in parts]
            if len(nums) % 3 != 0:
                raise ValueError(f"Line {i+1} in tract file does not have a 3N number of floats")
            pts = [(nums[j], nums[j+1], nums[j+2]) for j in range(0, len(nums), 3)]
            fibers.append(pts)
    return fibers

def load_valid_indices(path):
    """Load a list of indices from a JSON [0, 1, 3...] or text file (newlines)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Filter indices file not found: {path}")

    # Try JSON first
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [int(x) for x in data]
            elif isinstance(data, dict) and 'valid_inds' in data:
                return [int(x) for x in data['valid_inds']]
    except Exception:
        pass

    # Fallback to text lines
    indices = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                indices.append(int(float(parts[0])))
    return indices

def load_thresholds(path):
    with open(path) as f:
        data = json.load(f)
    thresholds = []
    for pw in pulse_widths:
        key = str(pw/1000)
        row = []
        if key in data:
            # Handle new format where thresholds are in dict or list form under the PW key
            pw_data = data[key]
            # Assumes keys are string numeric indices, find max key to determine list length or iterate
            # For robustness, we'll try to reconstruct based on available keys
            if isinstance(pw_data, dict):
                # Sort keys numerically to ensure order [0, 1, 2...]
                indices = sorted([int(k) for k in pw_data.keys()])
                for idx in indices:
                    row.append(pw_data[str(idx)])
            elif isinstance(pw_data, list):
                 row = pw_data
        thresholds.append(row)
    return thresholds

def render_scene_plotly(fibers, thresholds_row, voltage_limit, title='',
                        show_axes=False, electrode_config=None, field_data=None):
    """Render the scene using Plotly with toggleable legend entries."""

    fig = go.Figure()

    # Densify helper for smoother lines
    def densify_points(xs, ys, zs, factor=4):
        if factor <= 1:
            return xs, ys, zs
        new_xs, new_ys, new_zs = [], [], []
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[i], ys[i+1]
            z0, z1 = zs[i], zs[i+1]
            for t in np.linspace(0, 1, factor, endpoint=False):
                new_xs.append(x0 + (x1 - x0) * t)
                new_ys.append(y0 + (y1 - y0) * t)
                new_zs.append(z0 + (z1 - z0) * t)
        new_xs.append(xs[-1])
        new_ys.append(ys[-1])
        new_zs.append(zs[-1])
        return new_xs, new_ys, new_zs

    activated = []
    first_active = True
    first_inactive = True

    for i, fib in enumerate(fibers):
        thr = math.inf
        if i < len(thresholds_row) and thresholds_row[i] is not None:
            try:
                thr = float(thresholds_row[i])
            except Exception:
                thr = math.inf

        is_active = thr < voltage_limit
        color = 'red' if is_active else 'black'
        opacity = 1.0 if is_active else 0.15
        if is_active:
            activated.append(i)
            group = "activated"
            show = first_active
            first_active = False
        else:
            group = "inactive"
            show = first_inactive
            first_inactive = False

        xs = [p[0] for p in fib]
        ys = [p[1] for p in fib]
        zs = [p[2] for p in fib]
        xs, ys, zs = densify_points(xs, ys, zs, factor=4)

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=color, width=2),
            opacity=opacity,
            name="Activated" if is_active else "Inactive",
            legendgroup=group,
            showlegend=show,
            visible=True if is_active else "legendonly",
        ))

    # Electric field (replaces lead mesh)
    n_before_efield = len(fig.data)
    if field_data is not None:
        X, Y, Z, V = field_data
        add_electric_field(fig, X, Y, Z, V)

    # Scene axes
    show = show_axes
    axis_style = dict(visible=True, showticklabels=show, showgrid=show,
                      zeroline=show, title="" if not show else None)
    hide = dict(visible=False, showticklabels=False, showgrid=False, zeroline=False)

    scene_dict = dict(
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        aspectmode='data',
        xaxis=axis_style if show else hide,
        yaxis=axis_style if show else hide,
        zaxis=axis_style if show else hide,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=scene_dict,
        showlegend=True,
        legend=dict(itemsizing="constant", title="Click to toggle"),
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

    return fig, activated


def plot_activation_plotly(fibers, pulse_widths, thresholds, voltage_limit,
                           out_folder, interactive_pw=None, show_axes=False,
                           electrode_config=None, field_data=None):
    """Generate Plotly visualizations for fiber activation."""
    os.makedirs(out_folder, exist_ok=True)

    n_fibers = len(fibers)
    n_thr_rows = len(thresholds[0]) if thresholds else 0
    if n_thr_rows < n_fibers:
        print(f"Warning: thresholds provided for {n_thr_rows} fibers but tract has {n_fibers}")

    activation_summary = {}

    if interactive_pw is not None:
        pw_idx = int(interactive_pw)
        if pw_idx < 0 or pw_idx >= len(pulse_widths):
            raise ValueError(f"--interactive_pw {pw_idx} out of range [0, {len(pulse_widths)-1}]")

        pw = pulse_widths[pw_idx]
        row = thresholds[pw_idx] if pw_idx < len(thresholds) else []
        title = f"Pulse width: {pw} \u03bcs (index {pw_idx})"

        print(f"Rendering interactive view for PW index {pw_idx} (PW={pw})...")
        fig, activated = render_scene_plotly(
            fibers, row, voltage_limit,
            title=title, show_axes=show_axes,
            electrode_config=electrode_config, field_data=field_data,
        )

        out_html = os.path.join(out_folder, f"activation_pw_{pw_idx:02d}.html")
        fig.write_html(out_html)
        print(f"Saved interactive plot to {out_html}")

        activation_summary[str(pw)] = activated

    else:
        for pw_idx, pw in enumerate(pulse_widths):
            row = thresholds[pw_idx] if pw_idx < len(thresholds) else []
            title = f"Pulse width: {pw} \u03bcs (index {pw_idx})"

            print(f"Rendering PW index {pw_idx} (PW={pw})...")
            fig, activated = render_scene_plotly(
                fibers, row, voltage_limit,
                title=title, show_axes=show_axes,
                electrode_config=electrode_config, field_data=field_data,
            )

            out_html = os.path.join(out_folder, f"activation_pw_{pw_idx:02d}.html")
            fig.write_html(out_html)
            print(f"Saved {out_html} (activated: {len(activated)})")

            activation_summary[str(pw)] = activated

    # Write activation summary
    summary_path = os.path.join(out_folder, 'activation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'pulse_widths': pulse_widths,
            'voltage_limit': voltage_limit,
            'activated': activation_summary
        }, f, indent=2)
    print(f"Wrote activation summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot activation by pulse width using Plotly (interactive 3D in browser).")

    # Required named arguments
    parser.add_argument("--tract", required=True, dest="tract_file",
                        help="Path to the fiber tract file (.txt)")
    parser.add_argument("--results", required=True, dest="thresholds_json",
                        help="Path to the simulation results JSON file")
    parser.add_argument("--output", required=True, dest="out_folder",
                        help="Directory to save output images and HTML files")

    # Optional arguments with defaults
    parser.add_argument("--voltage", type=float, dest="voltage_limit", default=5.0,
                        help="Default threshold voltage (V) for activation. "
                             "Can be changed interactively in the HTML viewer. (default: 5.0)")
    parser.add_argument("--cond", choices=["anisotropic", "isotropic"],
                        dest="conductivity", default="anisotropic",
                        help="Conductivity type (default: anisotropic)")

    # Extra features
    parser.add_argument("--show_axes", action="store_true",
                        help="Show XYZ axes in the Plotly 3D scene (default: hidden)")
    parser.add_argument("--filter_indices", type=str, default=None,
                        help="Path to a text/JSON file listing the indices of fibers that were simulated.")
    parser.add_argument("--all_fibers", action="store_true",
                        help="Plot ALL fibers at full length, ignoring the filter_indices file. "
                             "Fibers without a threshold are drawn as inactive.")
    parser.add_argument("--electrode_config", type=str, default=None,
                        help="Electrode contact configuration, e.g. '01-23', '+012-3', '-0+1-2+3'. "
                             "'-' = cathode (red), '+' = anode (blue), unmarked = inactive (grey).")
    parser.add_argument("--electrode", type=str, default=None,
                        help="Path to the electrode FEM export file (.txt, COMSOL format). "
                             "Shows an electric-field isosurface instead of a lead mesh.")
    parser.add_argument("--electrode_center", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="X Y Z position of the electrode center for field re-centering.")
    parser.add_argument("--field_subsample", type=int, default=4, metavar="N",
                        help="Subsample the electrode grid by keeping every Nth point per axis. "
                             "Default: 4.")

    args = parser.parse_args()

    base_out = args.out_folder
    mkdirp(base_out)

    electrode_center = tuple(args.electrode_center) if args.electrode_center else (0, 0, 0)

    print(f"Reading {args.tract_file}...")
    fibers = read_tract_file(args.tract_file)
    print(f"Read {len(fibers)} fibers")

    if args.all_fibers:
        print(f"--all_fibers: showing all {len(fibers)} fibers (no FEM filter).")
    elif args.filter_indices:
        print(f"Loading filter indices from {args.filter_indices}...")
        valid_indices = load_valid_indices(args.filter_indices)
        try:
            filtered_fibers = [fibers[idx] for idx in valid_indices]
            print(f"Filtered fibers: kept {len(filtered_fibers)} of {len(fibers)} original fibers.")
            fibers = filtered_fibers
        except IndexError as e:
            print(f"Error: filter index {e} is out of bounds for the loaded tract file.")
            sys.exit(1)

    electrode_config = parse_electrode_config(args.electrode_config) if args.electrode_config else None

    # Load electric field if an electrode file was provided
    field_data = None
    if args.electrode:
        field_data = load_electrode_field(
            args.electrode,
            subsample=args.field_subsample,
            electrode_center=electrode_center,
        )

    print(f"Reading {args.thresholds_json}...")
    thresholds = load_thresholds(args.thresholds_json)
    if len(thresholds) != len(pulse_widths):
        print(f"Warning: expected {len(pulse_widths)} pulse widths but got {len(thresholds)}")

    # Prepare fiber data for embedding (flatten tuples to [x,y,z,x,y,z,...])
    print(f"\nPreparing fiber data...")
    fiber_coords = []
    for fib in fibers:
        flat = []
        for x, y, z in fib:
            flat.extend([round(x, 2), round(y, 2), round(z, 2)])
        fiber_coords.append(flat)

    # Collect thresholds per PW
    all_thresholds = {}
    pw_options = []
    for pw_idx, pw in enumerate(pulse_widths):
        pw_us = str(pw)
        row = thresholds[pw_idx] if pw_idx < len(thresholds) else []
        padded = []
        for i in range(len(fibers)):
            if i < len(row) and row[i] is not None:
                try:
                    padded.append(round(float(row[i]), 4))
                except (TypeError, ValueError):
                    padded.append(None)
            else:
                padded.append(None)
        all_thresholds[pw_us] = padded
        pw_options.append((pw_us, f"{pw} \u03bcs"))

    # E-field traces
    efield_json = prepare_efield_traces(field_data)

    # Generate single interactive HTML
    out_html = os.path.join(base_out, "activation.html")
    write_interactive_html(
        out_html, fiber_coords, all_thresholds, pw_options,
        efield_json, args.show_axes, default_voltage=args.voltage_limit,
    )
    print(f"\nDone. Interactive viewer -> {out_html}")


if __name__ == "__main__":
    main()