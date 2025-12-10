import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def load_waveforms(path: Path):
    # Try memmapped npy first, then pickle
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return arr


def interactive_view(waveforms):
    # waveforms shape: (N_samples, N_pulses, L)
    n_samples, n_pulses, L = waveforms.shape

    cur_sample = 0
    cur_pulse = 0

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)

    ax_heat = fig.add_subplot(gs[:, 0])  # left: heatmap (pulses x time)
    ax_overlay = fig.add_subplot(gs[0, 1])  # top-right: overlay
    ax_pulse = fig.add_subplot(gs[1, 1])  # bottom-right: single pulse

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)

    def draw():
        ax_heat.clear()
        ax_overlay.clear()
        ax_pulse.clear()

        w = waveforms[cur_sample]
        ax_heat.imshow(w, aspect="auto", cmap="viridis")
        ax_heat.set_title(
            f"Sample {cur_sample} heatmap ({n_pulses} pulses x {L} samples)"
        )
        ax_heat.set_xlabel("Time")
        ax_heat.set_ylabel("Pulse idx")

        # overlay
        for i in range(n_pulses):
            ax_overlay.plot(w[i], color="C0", alpha=0.12)
        ax_overlay.plot(w.mean(axis=0), color="red", linewidth=2, label="mean")
        ax_overlay.set_title(f"Sample {cur_sample} overlay")
        ax_overlay.set_xlabel("Time")
        ax_overlay.legend()

        # selected pulse
        ax_pulse.plot(w[cur_pulse], color="C2")
        ax_pulse.set_title(f"Sample {cur_sample} - Pulse {cur_pulse}")
        ax_pulse.set_xlabel("Time")

        fig.canvas.draw_idle()

    # initial draw
    draw()

    # Slider axes
    ax_s_sample = plt.axes([0.15, 0.05, 0.65, 0.03])
    ax_s_pulse = plt.axes([0.15, 0.01, 0.65, 0.03])
    s_sample = Slider(ax_s_sample, "Sample", 0, n_samples - 1, valinit=0, valstep=1)
    s_pulse = Slider(ax_s_pulse, "Pulse", 0, n_pulses - 1, valinit=0, valstep=1)

    def on_sample(val):
        nonlocal cur_sample
        cur_sample = int(val)
        draw()

    def on_pulse(val):
        nonlocal cur_pulse
        cur_pulse = int(val)
        draw()

    s_sample.on_changed(on_sample)
    s_pulse.on_changed(on_pulse)

    # Buttons for prev/next sample and pulse
    axprev_s = plt.axes([0.02, 0.85, 0.08, 0.04])
    axnext_s = plt.axes([0.02, 0.79, 0.08, 0.04])
    bprev_s = Button(axprev_s, "Prev Samp")
    bnext_s = Button(axnext_s, "Next Samp")

    axprev_p = plt.axes([0.86, 0.85, 0.08, 0.04])
    axnext_p = plt.axes([0.86, 0.79, 0.08, 0.04])
    bprev_p = Button(axprev_p, "Prev Pulse")
    bnext_p = Button(axnext_p, "Next Pulse")

    def prev_sample(event):
        nonlocal cur_sample
        cur_sample = max(0, cur_sample - 1)
        s_sample.set_val(cur_sample)

    def next_sample(event):
        nonlocal cur_sample
        cur_sample = min(n_samples - 1, cur_sample + 1)
        s_sample.set_val(cur_sample)

    def prev_pulse(event):
        nonlocal cur_pulse
        cur_pulse = max(0, cur_pulse - 1)
        s_pulse.set_val(cur_pulse)

    def next_pulse(event):
        nonlocal cur_pulse
        cur_pulse = min(n_pulses - 1, cur_pulse + 1)
        s_pulse.set_val(cur_pulse)

    bprev_s.on_clicked(prev_sample)
    bnext_s.on_clicked(next_sample)
    bprev_p.on_clicked(prev_pulse)
    bnext_p.on_clicked(next_pulse)

    # Keyboard bindings
    def on_key(event):
        nonlocal cur_sample, cur_pulse
        if event.key == "left":
            prev_pulse(event)
        elif event.key == "right":
            next_pulse(event)
        elif event.key == "up":
            prev_sample(event)
        elif event.key == "down":
            next_sample(event)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="E:\\Graduate\\projects\\multimodal_vsb_20251208\\research\\data\\",
        help="Directory containing the data files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    path = Path(args.data_dir) / "all_chunk_waves_160chunks.dat"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    waveforms = load_waveforms(path)
    # If memmap returned, ensure we can index as usual
    interactive_view(waveforms)
