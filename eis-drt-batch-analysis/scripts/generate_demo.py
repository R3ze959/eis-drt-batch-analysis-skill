#!/usr/bin/env python3
"""Generate reproducible SYNTHETIC EIS only; refuse nonempty destinations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def generate(output, nodes=4):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Demo output must be new/empty; no overwrite")
    if nodes < 2:
        raise ValueError("At least two nodes per direction")
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    f = np.logspace(5, -2, 61)
    w = 2*np.pi*f
    rows, truth = [], []
    for direction in ("charge", "discharge"):
        voltages = np.linspace(3.0, 3.7, nodes)
        if direction == "discharge":
            voltages = voltages[::-1]
        for i, v in enumerate(voltages):
            r = 3.0 + 5*(v-3.0)
            tau = 0.02 * (1 + 3*(v-3.0))
            if direction == "discharge":
                r *= 1.3
                tau *= 2
            z = 1.2 + 1j*w*2e-6 + 1.8/(1+1j*w*2e-4) + r/(1+1j*w*tau)
            name = f"{direction}_{i+1:02d}.csv"
            with (raw/name).open("w", encoding="utf-8", newline="") as h:
                writer = csv.writer(h)
                writer.writerow(["Frequency (Hz)", "Z' (ohm)", "-Z'' (ohm)"])
                writer.writerows(zip(f, z.real, -z.imag))
            rows.append({"path": "raw/"+name, "cell_id": "DEMO",
                         "label": f"Synthetic {direction} {v:.3f} V", "direction": direction,
                         "cycle": "1", "temperature_C": "25", "voltage_v": f"{v:.6f}",
                         "protocol_id": "synthetic-demo", "ac_amplitude_mv": "10",
                         "rest_time_s": "1800", "data_origin": "synthetic",
                         "input_role": "eis", "include": "true"})
            truth.append({"file": name, "R_inf_ohm": 1.2, "L_henry": 2e-6,
                          "RC": [{"R_ohm": 1.8, "tau_s": 2e-4}, {"R_ohm": r, "tau_s": tau}]})
    with (output/"manifest.csv").open("w", encoding="utf-8", newline="") as h:
        writer=csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"synthetic_truth.json").write_text(json.dumps({
        "data_origin": "synthetic", "no_experimental_data": True, "cases": truth}, indent=2), encoding="utf-8")
    return output/"manifest.csv"


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--nodes",type=int,default=4)
    a=p.parse_args()
    print(generate(a.output,a.nodes))
