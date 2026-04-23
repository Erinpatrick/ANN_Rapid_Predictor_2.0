"""
Fast 3D visualization with named tract bundles (DRTT, ML, PTR) highlighted.

Reads ANN prediction results, the corresponding tract file, and a manifest
JSON (from move_whole_brain_tracts.py) that records which fiber indices belong
to each named bundle.  Bundles are drawn in distinct colours, separate from
the generic whole-brain fibers, in both activated and inactive states.

Colour scheme
-------------
  Whole-brain activated   : red
  Whole-brain inactive    : black (transparent)
  DRTT activated          : lime green
  DRTT inactive           : dark green (transparent)
  ML activated            : cyan
  ML inactive             : teal (transparent)
  PTR activated           : magenta
  PTR inactive            : purple (transparent)
"""

import os
import sys
import json
import argparse
import numpy as np
import plotly.graph_objects as go
import time


# ---------------------------------------------------------------------------
# Bundle colour definitions
# ---------------------------------------------------------------------------

# Each entry: (activated_color, inactive_color, activated_opacity, inactive_opacity)
BUNDLE_STYLES = {
    "L_DRTT_voxel": ("#00FF00", "#006400", 1.0, 0.35),   # lime / dark green
    "L_ML_voxel":   ("#00FFFF", "#008080", 1.0, 0.35),   # cyan / teal
    "L_PTR_voxel":  ("#FF00FF", "#800080", 1.0, 0.35),   # magenta / purple
}

# Fallback for unrecognised bundles
DEFAULT_BUNDLE_STYLE = ("#FFA500", "#8B4500", 1.0, 0.35)  # orange / dark orange

WHOLE_BRAIN_ACTIVE   = ("red",   1.0)
WHOLE_BRAIN_INACTIVE = ("black", 0.15)


# ---------------------------------------------------------------------------
# Utility helpers (shared with plot_tracts_fast.py)
# ---------------------------------------------------------------------------

def mkdirp(path):
    os.makedirs(path, exist_ok=True)


def read_tract_file(path, downsample=1):
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
    with open(path) as f:
        data = json.load(f)
    valid_inds = data.get("valid_inds", [])
    pw_keys = []
    for k in data:
        try:
            pw_keys.append(float(k))
        except ValueError:
            pass
    pw_keys.sort()
    return data, pw_keys, [int(i) for i in valid_inds]



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
        midpoint sits at these coordinates.

    Returns
    -------
    X, Y, Z : 3-D ndarrays (meshgrid)
    V : 3-D ndarray of electric potential
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


def get_thresholds_for_pw(data, pw_key):
    """Return a dict mapping original fiber index -> threshold value."""
    pw_data = data.get(str(pw_key), data.get(pw_key, {}))
    if isinstance(pw_data, list):
        return {i: v for i, v in enumerate(pw_data)}
    if isinstance(pw_data, dict):
        return {int(k): v for k, v in pw_data.items()}
    return {}


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Interactive HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE_BUNDLES = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fiber Activation Viewer (Bundles)</title>
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
var BUNDLE_RANGES = ___BUNDLE_RANGES___;
var BUNDLE_STYLES = ___BUNDLE_STYLES___;
var WB_SIMPLIFY = ___WB_SIMPLIFY___;
var V_TO_MA = 1.123;

function syncMA(vInput) {
  var v = parseFloat(vInput.value);
  if (!isNaN(v)) document.getElementById("ma-input").value = (v * V_TO_MA).toFixed(3);
}
function syncV(maInput) {
  var ma = parseFloat(maInput.value);
  if (!isNaN(ma)) document.getElementById("voltage-input").value = (ma / V_TO_MA).toFixed(4);
}

function buildMergedTrace(indices, color, width, opacity, name, visible, legendgroup, stride) {
  stride = stride || 1;
  var x = [], y = [], z = [];
  for (var ii = 0; ii < indices.length; ii++) {
    var fib = FIBERS[indices[ii]];
    for (var j = 0; j < fib.length; j += 3 * stride) {
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
    var bundleNames = Object.keys(BUNDLE_RANGES);
    var fiberBundle = {};
    for (var bi = 0; bi < bundleNames.length; bi++) {
      var binfo = BUNDLE_RANGES[bundleNames[bi]];
      for (var idx = binfo.start; idx < binfo.end; idx++) fiberBundle[idx] = bundleNames[bi];
    }
    var wbActive = [], wbInactive = [];
    var bundleActive = {}, bundleInactive = {};
    for (var bi = 0; bi < bundleNames.length; bi++) {
      bundleActive[bundleNames[bi]] = [];
      bundleInactive[bundleNames[bi]] = [];
    }
    for (var i = 0; i < FIBERS.length; i++) {
      var thr = (i < thresholds.length && thresholds[i] !== null) ? thresholds[i] : Infinity;
      var act = thr < voltage;
      if (fiberBundle[i] !== undefined) {
        (act ? bundleActive : bundleInactive)[fiberBundle[i]].push(i);
      } else {
        (act ? wbActive : wbInactive).push(i);
      }
    }
    var traces = [];
    if (wbActive.length > 0)
      traces.push(buildMergedTrace(wbActive, "red", 2, 1.0, "Whole-brain activated", true, "wb_activated", WB_SIMPLIFY));
    if (wbInactive.length > 0)
      traces.push(buildMergedTrace(wbInactive, "black", 1, 0.15, "Whole-brain inactive", "legendonly", "wb_inactive", WB_SIMPLIFY));
    for (var bi = 0; bi < bundleNames.length; bi++) {
      var bn = bundleNames[bi];
      var st = BUNDLE_STYLES[bn] || {actColor:"#FFA500",inactColor:"#8B4500",actOpacity:1.0,inactOpacity:0.35};
      var dn = bn.replace(/^L_/, "").replace(/_voxel$/, "");
      if (bundleActive[bn].length > 0)
        traces.push(buildMergedTrace(bundleActive[bn], st.actColor, 3, st.actOpacity, dn+" activated", true, bn+"_act", 1));
      if (bundleInactive[bn].length > 0)
        traces.push(buildMergedTrace(bundleInactive[bn], st.inactColor, 2, st.inactOpacity, dn+" inactive", "legendonly", bn+"_inact", 1));
    }
    var efStart = traces.length;
    for (var k = 0; k < EFIELD_TRACES.length; k++) {
      var tc = {}; for (var key in EFIELD_TRACES[k]) tc[key] = EFIELD_TRACES[k][key]; traces.push(tc);
    }
    var axStyle = SHOW_AXES
      ? {visible:true, showticklabels:true, showgrid:true, zeroline:true, title:""}
      : {visible:false, showticklabels:false, showgrid:false, zeroline:false};
    var layout = {
      title: {text: "Pulse Width: "+pwKey+" \u03bcs  |  Threshold: "+voltage.toFixed(4)+" V  /  "+(voltage * V_TO_MA).toFixed(3)+" mA", x:0.5, xanchor:"center"},
      scene: {camera:{eye:{x:1.5,y:1.5,z:1.5}}, aspectmode:"data", xaxis:axStyle, yaxis:axStyle, zaxis:axStyle},
      showlegend: true,
      legend: {itemsizing:"constant", title:{text:"Click to toggle"}},
      margin: {l:60, r:0, t:30, b:0}
    };
    if (EFIELD_TRACES.length > 0) {
      var efIdxs = [];
      for (var ei = efStart; ei < traces.length; ei++) efIdxs.push(ei);
      layout.updatemenus = [{
        type:"buttons", buttons:[{
          label:"Toggle Electric Field", method:"restyle",
          args:[{visible:efIdxs.map(function(){return false;})},efIdxs],
          args2:[{visible:efIdxs.map(function(){return true;})},efIdxs]
        }], showactive:true, x:0.0, xanchor:"left", y:1.05, yanchor:"top"
      }];
    }
    for (var t = 0; t < traces.length; t++) {
      var lg = traces[t].legendgroup;
      if (lg && savedVis[lg] !== undefined) traces[t].visible = savedVis[lg];
    }
    if (savedCamera) layout.scene.camera = savedCamera;
    Plotly.react("plot", traces, layout);
    var totalActive = wbActive.length;
    for (var bi = 0; bi < bundleNames.length; bi++) totalActive += bundleActive[bundleNames[bi]].length;
    document.getElementById("status").textContent = "Activated: "+totalActive+" / "+FIBERS.length+" fibers";
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
</html>"""  # end HTML_TEMPLATE_BUNDLES


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
                            efield_traces_json, show_axes, bundle_ranges,
                            bundle_styles_js, wb_simplify, default_voltage=5.0):
    """Write a single interactive HTML viewer with bundle-aware colouring."""
    opts_html = "\n".join(
        f'      <option value="{key}">{label}</option>' for key, label in pw_options
    )
    html = HTML_TEMPLATE_BUNDLES
    html = html.replace("___PW_OPTIONS___", opts_html)
    html = html.replace("___FIBERS_DATA___", json.dumps(fiber_coords))
    html = html.replace("___THRESHOLDS_DATA___", json.dumps(all_thresholds))
    html = html.replace("___EFIELD_DATA___", efield_traces_json)
    html = html.replace("___SHOW_AXES___", "true" if show_axes else "false")
    html = html.replace("___BUNDLE_RANGES___", json.dumps(bundle_ranges))
    html = html.replace("___BUNDLE_STYLES___", json.dumps(bundle_styles_js))
    html = html.replace("___WB_SIMPLIFY___", str(wb_simplify))
    html = html.replace("___DEFAULT_VOLTAGE___", str(default_voltage))
    html = html.replace("___DEFAULT_MA___", str(round(default_voltage * 1.123, 3)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote interactive HTML: {path}")


# ---------------------------------------------------------------------------
# Batched trace building (kept for backward compatibility)
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
# Electrode (copied from plot_tracts_fast.py)
# ---------------------------------------------------------------------------

def parse_electrode_config(config_str):
    contact_polarity = {0: None, 1: None, 2: None, 3: None}
    current_sign = None
    for ch in config_str:
        if ch in '+-':
            current_sign = ch
        elif ch.isdigit():
            contact_polarity[int(ch)] = current_sign
            current_sign = None
    return contact_polarity



# ---------------------------------------------------------------------------
# Scene rendering (bundle-aware)
# ---------------------------------------------------------------------------

def render(fibers, threshold_map, voltage, electrode_center, bundle_ranges,
           title="", show_axes=False, electrode_config=None,
           fiber_index_map=None, field_data=None, wb_simplify=3):
    """Build a Plotly figure with bundle-aware colouring.

    Parameters
    ----------
    fibers : list of ndarray
        Fiber coordinate arrays.
    threshold_map : dict
        Mapping original_fiber_index -> threshold value.
    voltage : float
        Activation threshold.
    bundle_ranges : dict
        Mapping bundle_name -> {"start": int, "end": int}.
        Fiber indices in [start, end) belong to that bundle.
    fiber_index_map : list or None
        Maps display index i to original tract file line number.
        If None, identity mapping (i -> i) is used.
    field_data : tuple or None
        (X, Y, Z, V) 3-D arrays from load_electrode_field().
    """
    fig = go.Figure()

    # Build a set of all bundle indices for fast lookup
    bundle_indices = {}   # bundle_name -> set of indices
    all_bundle_idxs = set()
    for bname, binfo in bundle_ranges.items():
        s, e = binfo["start"], binfo["end"]
        idxs = set(range(s, e))
        bundle_indices[bname] = idxs
        all_bundle_idxs |= idxs

    # Classify every fiber
    wb_active, wb_inactive = [], []
    bundle_active = {b: [] for b in bundle_ranges}
    bundle_inactive = {b: [] for b in bundle_ranges}

    for i in range(len(fibers)):
        # Look up the original tract file line number for this display fiber
        orig_idx = fiber_index_map[i] if fiber_index_map is not None else i
        thr = float("inf")
        if orig_idx in threshold_map:
            try:
                thr = float(threshold_map[orig_idx])
            except (TypeError, ValueError):
                pass
        activated = thr < voltage

        # Check if this fiber belongs to a bundle
        in_bundle = False
        for bname, idxs in bundle_indices.items():
            if i in idxs:
                (bundle_active[bname] if activated else bundle_inactive[bname]).append(i)
                in_bundle = True
                break
        if not in_bundle:
            (wb_active if activated else wb_inactive).append(i)



    # Stats
    total_active = len(wb_active) + sum(len(v) for v in bundle_active.values())
    total_inactive = len(wb_inactive) + sum(len(v) for v in bundle_inactive.values())
    print(f"  Total activated: {total_active},  Total inactive: {total_inactive}")
    print(f"    Whole-brain: {len(wb_active)} activated, {len(wb_inactive)} inactive")
    for bname in bundle_ranges:
        print(f"    {bname}: {len(bundle_active[bname])} activated, "
              f"{len(bundle_inactive[bname])} inactive")

    # ── Whole-brain traces (simplified for performance) ─────────────
    stride = max(1, wb_simplify)
    if stride > 1:
        print(f"  Simplifying whole-brain fibers: keeping every {stride}th point")

    ax, ay, az = build_merged_trace(fibers, wb_active, pts_stride=stride)
    if ax is not None:
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az, mode="lines",
            line=dict(color=WHOLE_BRAIN_ACTIVE[0], width=2),
            opacity=WHOLE_BRAIN_ACTIVE[1],
            name="Whole-brain activated", connectgaps=False,
            legendgroup="wb_activated", showlegend=True,
        ))

    ix, iy, iz = build_merged_trace(fibers, wb_inactive, pts_stride=stride)
    if ix is not None:
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode="lines",
            line=dict(color=WHOLE_BRAIN_INACTIVE[0], width=1),
            opacity=WHOLE_BRAIN_INACTIVE[1],
            name="Whole-brain inactive", connectgaps=False,
            legendgroup="wb_inactive", showlegend=True,
            visible="legendonly",
        ))

    # ── Bundle traces ───────────────────────────────────────────────
    for bname in bundle_ranges:
        style = BUNDLE_STYLES.get(bname, DEFAULT_BUNDLE_STYLE)
        act_color, inact_color, act_opacity, inact_opacity = style
        display_name = bname.removeprefix("L_").removesuffix("_voxel")

        bx, by, bz = build_merged_trace(fibers, bundle_active[bname])
        if bx is not None:
            fig.add_trace(go.Scatter3d(
                x=bx, y=by, z=bz, mode="lines",
                line=dict(color=act_color, width=3),
                opacity=act_opacity,
                name=f"{display_name} activated", connectgaps=False,
                legendgroup=f"{bname}_act", showlegend=True,
            ))

        bx, by, bz = build_merged_trace(fibers, bundle_inactive[bname])
        if bx is not None:
            fig.add_trace(go.Scatter3d(
                x=bx, y=by, z=bz, mode="lines",
                line=dict(color=inact_color, width=2),
                opacity=inact_opacity,
                name=f"{display_name} inactive", connectgaps=False,
                legendgroup=f"{bname}_inact", showlegend=True,
                visible="legendonly",
            ))

    # ── Electric Field ──────────────────────────────────────────────
    n_before_efield = len(fig.data)
    if field_data is not None:
        X, Y, Z, V = field_data
        add_electric_field(fig, X, Y, Z, V)

    # ── Layout ──────────────────────────────────────────────────────
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

    # Build per-bundle activation detail (0-based indices within each bundle)
    bundle_detail = {}
    for bname, binfo in bundle_ranges.items():
        display_name = bname.removeprefix("L_").removesuffix("_voxel")
        start = binfo["start"]
        bundle_detail[display_name] = {
            "activated": len(bundle_active[bname]),
            "total": binfo["end"] - start,
            "activated_indices": sorted(i - start for i in bundle_active[bname]),
        }
    bundle_detail["whole_brain"] = {
        "activated": len(wb_active),
        "total": len(wb_active) + len(wb_inactive),
        "activated_indices": sorted(wb_active),
    }

    return fig, total_active, bundle_detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast 3D fiber visualizer with named bundle highlighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python plot_tracts_bundles.py \\
      --tract moved_whole_brain_tracts.txt \\
      --results results.json \\
      --manifest moved_whole_brain_tracts_manifest.json \\
      --output plots/ \\
      --electrode_center 167 213 147 \\
      --electrode_config 01-23 \\
      --all_fibers --show_axes
""",
    )

    parser.add_argument("--tract", required=True, metavar="FILE",
                        help="Path to the merged tract file (whole-brain + bundles).")
    parser.add_argument("--results", required=True, metavar="FILE",
                        help="Path to the JSON results file from dti_ann_LUT.py.")
    parser.add_argument("--manifest", required=True, metavar="FILE",
                        help="Path to the manifest JSON from move_whole_brain_tracts.py.")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="Directory where HTML plots will be saved.")
    parser.add_argument("--activation_threshold", type=float, default=5.0, metavar="VOLTS",
                        help="Default voltage (V) below which a fiber is activated. "
                             "Can be changed interactively in the HTML viewer. Default: 5.0")
    parser.add_argument("--electrode_center", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="X Y Z electrode centre for display.")
    parser.add_argument("--downsample", type=int, default=1, metavar="N",
                        help="Keep every Nth point per fiber. Default: 1")
    parser.add_argument("--wb_simplify", type=int, default=3, metavar="N",
                        help="Keep every Nth point of whole-brain fibers to reduce "
                             "rendering load. Bundle fibers are always full resolution. "
                             "Default: 3. Set to 1 for no simplification.")
    parser.add_argument("--show_axes", action="store_true",
                        help="Show X/Y/Z axes in the 3D scene.")
    parser.add_argument("--all_fibers", action="store_true",
                        help="Plot ALL fibers, ignoring the valid_inds filter.")
    parser.add_argument("--electrode_config", type=str, default=None, metavar="CONFIG",
                        help="Electrode config string, e.g. '01-23'. "
                             "'-' marks cathodes, '+' marks anodes.")
    parser.add_argument("--electrode", type=str, default=None, metavar="FILE",
                        help="Path to the electrode FEM export file (.txt, COMSOL format). "
                             "When provided, an electric-field isosurface is shown. "
                             "Toggle it on/off in the legend.")
    parser.add_argument("--field_subsample", type=int, default=4, metavar="N",
                        help="Subsample the electrode grid by keeping every Nth point per axis. "
                             "Default: 4.  Increase to reduce memory / rendering time.")

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

    # 1. Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    bundle_ranges = manifest.get("bundles", {})
    print(f"  Bundles: {list(bundle_ranges.keys())}")
    for bname, binfo in bundle_ranges.items():
        print(f"    {bname}: indices {binfo['start']}–{binfo['end'] - 1} ({binfo['count']} fibers)")

    # 2. Load results
    print(f"Loading results: {args.results}")
    data, pw_keys, valid_inds = load_results(args.results)
    if not pw_keys:
        print("ERROR: No pulse-width data found.")
        sys.exit(1)
    print(f"Found {len(pw_keys)} pulse width(s): " +
          ", ".join(f"{pw*1000:.0f} us" for pw in pw_keys))

    # 3. Read fibers
    fibers = read_tract_file(args.tract, downsample=args.downsample)

    # 4. Filter
    fiber_index_map = None  # maps display index -> original tract line number
    if args.all_fibers:
        print(f"--all_fibers: showing all {len(fibers)} fibers (no FEM filter).")
        # Identity mapping: display index i = original line i
    elif valid_inds:
        print(f"Filtering to {len(valid_inds)} valid fibers (from results file)...")
        try:
            fibers = [fibers[i] for i in valid_inds]
        except IndexError:
            print("ERROR: valid_inds references indices that don't exist in the tract file.")
            sys.exit(1)
        # Build mapping: display index j -> original line number valid_inds[j]
        fiber_index_map = valid_inds
        # Remap bundle ranges to the filtered index space
        vi_set = set(valid_inds)
        vi_map = {orig: new for new, orig in enumerate(valid_inds)}
        new_bundle_ranges = {}
        for bname, binfo in bundle_ranges.items():
            new_idxs = [vi_map[i] for i in range(binfo["start"], binfo["end"]) if i in vi_set]
            if new_idxs:
                new_bundle_ranges[bname] = {"start": min(new_idxs), "end": max(new_idxs) + 1}
            else:
                new_bundle_ranges[bname] = {"start": 0, "end": 0}
        bundle_ranges = new_bundle_ranges
        print(f"  {len(fibers)} fibers after filtering.")

    # 5. Prepare data for interactive HTML
    print(f"\nPreparing fiber data...")
    fiber_coords = []
    for fib in fibers:
        fiber_coords.append(np.round(fib, 2).flatten().tolist())

    # Collect thresholds per PW (remapped to display fiber indices)
    all_thresholds = {}
    pw_options = []
    for pw in pw_keys:
        pw_us = f"{pw * 1000:.0f}"
        thr_map = get_thresholds_for_pw(data, pw)
        thr_array = []
        for i in range(len(fibers)):
            orig_idx = fiber_index_map[i] if fiber_index_map is not None else i
            val = thr_map.get(orig_idx, None)
            if val is not None:
                try:
                    thr_array.append(round(float(val), 4))
                except (TypeError, ValueError):
                    thr_array.append(None)
            else:
                thr_array.append(None)
        all_thresholds[pw_us] = thr_array
        pw_options.append((pw_us, f"{pw_us} \u03bcs"))

    # E-field traces
    efield_json = prepare_efield_traces(field_data)

    # Convert bundle styles to JS-friendly dict
    bundle_styles_js = {}
    for bname in bundle_ranges:
        style = BUNDLE_STYLES.get(bname, DEFAULT_BUNDLE_STYLE)
        bundle_styles_js[bname] = {
            "actColor": style[0], "inactColor": style[1],
            "actOpacity": style[2], "inactOpacity": style[3],
        }

    # Generate single interactive HTML
    out_html = os.path.join(args.output, "activation.html")
    write_interactive_html(
        out_html, fiber_coords, all_thresholds, pw_options,
        efield_json, args.show_axes, bundle_ranges,
        bundle_styles_js, args.wb_simplify,
        default_voltage=args.activation_threshold,
    )
    print(f"\nDone. Interactive viewer -> {out_html}")


if __name__ == "__main__":
    main()
