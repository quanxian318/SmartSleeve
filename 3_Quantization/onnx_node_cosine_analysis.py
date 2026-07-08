#!/usr/bin/env python3
"""
ONNX Model Quantization Consistency Analysis
==============================================
Compare FP32 vs INT8 quantized model at EVERY intermediate node.
Generates cosine similarity curve for quantization error diagnosis.

Works with:
  - original_float_model.onnx (FP32 before optimization)
  - optimized_float_model.onnx (FP32 after optimization)
  - quantized_model.onnx (INT8 PTQ model)
  - test_input.npy (calibration/test data)

Output:
  - cosine_similarity_nodes.csv
  - cosine_similarity_nodes.json
  - cosine_similarity_nodes.png

For .bin BPU models: see section "BPU Intermediate Dump" below.
"""

import os, sys, io, csv, json, warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import onnx
import onnxruntime as ort
from collections import Counter

# ============================================================
# CONFIG — modify these paths
# ============================================================
FP32_ONNX    = "original_float_model.onnx"       # or optimized_float_model.onnx
QUANT_ONNX   = "quantized_model.onnx"             # PTQ quantized model
TEST_INPUT   = "test_input.npy"                   # .npy file with test data
OUTPUT_DIR   = "./quant_analysis_output"
N_SAMPLES    = 100                                # use first N samples from test data
EPS          = 1e-8
DPI          = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper: expand ONNX graph to output ALL intermediate tensors
# ============================================================
def add_all_intermediate_outputs(onnx_path, output_path):
    """Modify ONNX graph so every intermediate tensor becomes a graph output."""
    model = onnx.load(onnx_path)
    graph = model.graph

    # Collect existing output names
    existing_outputs = {o.name for o in graph.output}
    # Collect initializer names (constants - skip these)
    init_names = {i.name for i in graph.initializer}
    # Collect graph input names (skip these too)
    input_names = {i.name for i in graph.input}

    # Collect all intermediate tensor names from node inputs/outputs
    all_names = set()
    for node in graph.node:
        for name in node.input:
            if name and name not in init_names:
                all_names.add(name)
        for name in node.output:
            if name:
                all_names.add(name)

    added = 0
    for name in sorted(all_names):
        if name in existing_outputs or name in input_names:
            continue
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        graph.output.append(vi)
        added += 1

    onnx.save(model, output_path)
    return added, list(graph.output)


# ============================================================
# Helper: run ONNX inference and collect ALL outputs
# ============================================================
def run_onnx_all_outputs(onnx_path, input_data, input_name):
    """Run ONNX model and return dict of all output tensors."""
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, {input_name: input_data})
    return dict(zip(output_names, results))


# ============================================================
# Cosine similarity
# ============================================================
def cosine_sim(a, b):
    """Cosine similarity between two flat vectors."""
    if a is None or b is None:
        return float('nan')
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    if len(af) == 0 or len(bf) == 0:
        return float('nan')
    dot = np.dot(af, bf)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    return float(dot / (na * nb + EPS))


# ============================================================
# ONNX op type mapping (for labeling nodes)
# ============================================================
def get_op_type_map(onnx_path):
    """Return {output_name: op_type} mapping."""
    model = onnx.load(onnx_path)
    op_map = {}
    for node in model.graph.node:
        for out in node.output:
            op_map[out] = node.op_type
    return op_map


# ============================================================
# For .bin models: BPU intermediate dump instructions
# ============================================================
BPU_DUMP_DOC = """
# ============================================================
# BPU .bin Model Intermediate Dump (RDK X5)
# ============================================================
# To dump intermediate layer outputs from a .bin BPU model:
#
#   hrt_model_exec infer \\
#     --model_file anchorcalib_tcn_bpu_v2.bin \\
#     --input_file test_frame.bin \\
#     --dump_intermediate 1 \\
#     --dump_format txt
#
# This generates per-layer dump files. Parse them with:
#   import numpy as np
#   layer_output = np.loadtxt("layer_0_output.txt")
#
# Then run this script with the parsed numpy arrays as the
# quantized model outputs dict.
"""


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  ONNX Quantization Node-by-Node Cosine Analysis")
    print("=" * 65)

    # ---- 1. Prepare expanded ONNX models ----
    print("\n[1/5] Expanding ONNX models to output all intermediate tensors...")
    fp32_expanded = os.path.join(OUTPUT_DIR, "_fp32_allnodes.onnx")
    qu8_expanded  = os.path.join(OUTPUT_DIR, "_quant_allnodes.onnx")

    n_fp32, _ = add_all_intermediate_outputs(FP32_ONNX, fp32_expanded)
    n_qu8, _  = add_all_intermediate_outputs(QUANT_ONNX, qu8_expanded)
    print(f"  FP32 intermediate outputs added: {n_fp32}")
    print(f"  INT8 intermediate outputs added: {n_qu8}")

    # ---- 2. Load test data ----
    print(f"\n[2/5] Loading test data: {TEST_INPUT}")
    x_test = np.load(TEST_INPUT)
    n_total = len(x_test)
    n_use = min(N_SAMPLES, n_total)
    x_test = x_test[:n_use]
    # Handle batch dimension: model expects [1, 26, 1, 64]
    if x_test.ndim == 4:  # [N, C, H, W]
        pass  # already batched
    print(f"  Using {n_use}/{n_total} samples, shape={x_test.shape}")

    # ---- 3. Run FP32 inference (sample by sample) ----
    print("\n[3/5] Running FP32 inference with all intermediate outputs...")
    fp32_iname = onnx.load(fp32_expanded).graph.input[0].name
    qu8_iname  = onnx.load(qu8_expanded).graph.input[0].name

    # Collect per-sample outputs; average cosine across samples
    fp32_all_outputs = {}
    qu8_all_outputs = {}

    for i in range(n_use):
        inp = x_test[i:i+1] if x_test.ndim == 4 else x_test[i:i+1]
        if x_test.ndim == 3:
            inp = inp.reshape(1, *x_test.shape[1:])

        try:
            fp32_out = run_onnx_all_outputs(fp32_expanded, inp.astype(np.float32), fp32_iname)
            qu8_out  = run_onnx_all_outputs(qu8_expanded, inp.astype(np.float32), qu8_iname)

            for k, v in fp32_out.items():
                fp32_all_outputs.setdefault(k, []).append(v)
            for k, v in qu8_out.items():
                qu8_all_outputs.setdefault(k, []).append(v)
        except Exception as e:
            print(f"  [WARN] Sample {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_use} samples done")

    print(f"  FP32 outputs: {len(fp32_all_outputs)} tensors")
    print(f"  INT8 outputs: {len(qu8_all_outputs)} tensors")

    # ---- 4. Compute per-node cosine similarity ----
    print("\n[4/5] Computing per-node cosine similarity...")

    op_map = get_op_type_map(FP32_ONNX)
    common_names = sorted(set(fp32_all_outputs.keys()) & set(qu8_all_outputs.keys()))

    # Filter: skip initializers and graph inputs
    fp32_model = onnx.load(fp32_expanded)
    init_names = {i.name for i in fp32_model.graph.initializer}
    input_names = {i.name for i in fp32_model.graph.input}

    results = []
    cosine_values = []
    skipped = {'init': 0, 'input': 0, 'shape_mismatch': 0, 'nan': 0}

    for idx, name in enumerate(common_names):
        if name in init_names:
            skipped['init'] += 1; continue
        if name in input_names:
            skipped['input'] += 1; continue

        fp32_vals = fp32_all_outputs[name]
        qu8_vals  = qu8_all_outputs[name]

        # Average cosine across samples
        cos_per_sample = []
        for fp32_t, qu8_t in zip(fp32_vals, qu8_vals):
            if fp32_t.shape != qu8_t.shape:
                skipped['shape_mismatch'] += 1
                cos_per_sample.append(float('nan'))
                continue
            cos_per_sample.append(cosine_sim(fp32_t, qu8_t))

        cos_mean = float(np.nanmean(cos_per_sample)) if cos_per_sample else float('nan')
        op_type = op_map.get(name, 'unknown')

        results.append({
            'node_index': len(results),
            'node_name': name,
            'op_type': op_type,
            'cosine_similarity': cos_mean,
            'fp32_shape': str(tuple(fp32_vals[0].shape)),
        })

        if not np.isnan(cos_mean):
            cosine_values.append(cos_mean)

        if idx < 10 or idx % 50 == 0:
            print(f"  [{len(results):4d}] {name[:45]:45s} op={op_type:<12s} cos={cos_mean:.6f}")

    # ---- 5. Statistics & Output ----
    print("\n[5/5] Statistics & Output...")
    cosine_arr = np.array(cosine_values)
    mean_cos = float(np.mean(cosine_arr))
    std_cos  = float(np.std(cosine_arr))
    min_cos  = float(np.min(cosine_arr))
    max_cos  = float(np.max(cosine_arr))
    below_098 = int(np.sum(cosine_arr < 0.98))
    below_095 = int(np.sum(cosine_arr < 0.95))

    sorted_idx = np.argsort(cosine_arr)
    worst_10 = [(results[i]['node_name'], cosine_arr[i])
                for i in sorted_idx[:10] if i < len(results)]

    print(f"\n  Total intermediate tensors:    {len(common_names)}")
    print(f"  Valid cosine values:           {len(cosine_values)}")
    print(f"  Skipped: init={skipped['init']} input={skipped['input']} "
          f"shape={skipped['shape_mismatch']}")
    print(f"  Mean cosine:    {mean_cos:.6f}")
    print(f"  Std cosine:     {std_cos:.6f}")
    print(f"  Min cosine:     {min_cos:.6f}")
    print(f"  Max cosine:     {max_cos:.6f}")
    print(f"  Cosine < 0.98:  {below_098}/{len(cosine_values)}")
    print(f"  Cosine < 0.95:  {below_095}/{len(cosine_values)}")
    print(f"\n  Worst 10 nodes:")
    for i, (name, val) in enumerate(worst_10):
        op = op_map.get(name, '?')
        print(f"    {i+1}. [{op}] {name[:55]}  cos={val:.6f}")

    # --- CSV ---
    csv_path = os.path.join(OUTPUT_DIR, 'cosine_similarity_nodes.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['node_index','node_name','op_type',
                                           'cosine_similarity','fp32_shape'])
        w.writeheader(); w.writerows(results)
    print(f"\n  [OK] {csv_path}")

    # --- JSON ---
    json_path = os.path.join(OUTPUT_DIR, 'cosine_similarity_nodes.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'AnchorCalibTCN_BPU',
            'comparison': 'FP32_vs_INT8_quantized',
            'n_samples': n_use,
            'num_tensors_total': len(common_names),
            'num_tensors_valid': len(cosine_values),
            'mean_cosine': mean_cos, 'std_cosine': std_cos,
            'min_cosine': min_cos, 'max_cosine': max_cos,
            'below_0.98': below_098, 'below_0.95': below_095,
            'skipped': skipped,
            'worst_10': [{'name': n, 'cosine': float(v)} for n, v in worst_10],
        }, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {json_path}")

    # --- PNG ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # Font setup
    cn = [f for f in fm.findSystemFonts()
          if any(k in fm.FontProperties(fname=f).get_name().lower()
                 for k in ['yahei','simhei','microsoft yahei'])]
    plt.rcParams['font.sans-serif'] = cn + ['DejaVu Sans'] if cn else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(22, 8))
    x = np.arange(len(results))
    y = np.array([r['cosine_similarity'] for r in results])
    valid = ~np.isnan(y)
    xp, yp = x[valid], y[valid]

    ax.fill_between(xp, yp - 0.0005, yp + 0.0005, alpha=0.2, color='#E8A0BF')
    ax.plot(xp, yp, '-', color='#D6336C', linewidth=1.6, alpha=0.95, label='all', zorder=3)

    # Annotate minimum
    wi = xp[np.argmin(yp)]
    wv = yp.min()
    ax.annotate(f'{wv:.5f}', xy=(wi, wv), xytext=(wi + 2, wv - 0.015),
                ha='left', fontsize=10, fontweight='bold', color='#D6336C',
                arrowprops=dict(arrowstyle='->', color='#D6336C', lw=1.5))

    ax.axhline(y=0.99, color='gray', linestyle=':', linewidth=1.2, alpha=0.6,
               label='0.99 threshold')
    ax.set_xlabel('Node Index', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('non_quantize_node_accumulate_err_of_node', fontsize=16, fontweight='bold')
    ax.set_ylim(0.90, 1.005)
    ax.set_xlim(-2, len(results) + 2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=12)

    stats_txt = (f'Total Nodes: {len(results)}\nMean Cos:   {mean_cos:.6f}\n'
                 f'Std Cos:    {std_cos:.6f}\nMin Cos:    {min_cos:.6f}\n'
                 f'Max Cos:    {max_cos:.6f}\n< 0.98:     {below_098}/{len(results)}\n'
                 f'< 0.95:     {below_095}/{len(results)}')
    ax.text(0.985, 0.97, stats_txt, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'cosine_similarity_nodes.png')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {png_path}")

    print(f"\n{'='*65}")
    print(f"  Done! Output: {OUTPUT_DIR}/")
    print(f"  cosine_similarity_nodes.csv")
    print(f"  cosine_similarity_nodes.json")
    print(f"  cosine_similarity_nodes.png")
    print(f"{'='*65}")

    # Print BPU dump instructions
    print(BPU_DUMP_DOC)


if __name__ == '__main__':
    main()
