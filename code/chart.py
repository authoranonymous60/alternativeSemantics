import matplotlib.pyplot as plt

# --------------------------------------------------
# Data
# --------------------------------------------------
data = [
    # Model, FS, FH, Inf   (FH = focus hint; "y" means present)
    ("G4P", 0, "n", 0.44),
    ("G4P", 0, "y", 0.50),
    ("G4P", 2, "n", 0.49),
    ("G4P", 2, "y", 0.50),
    ("G4P", 5, "n", 0.38),
    ("G4P", 5, "y", 0.36),

    ("GPA", 0, "n", 0.44),
    ("GPA", 0, "y", 0.44),
    ("GPA", 2, "n", 0.47),
    ("GPA", 2, "y", 0.48),
    ("GPA", 5, "n", 0.35),
    ("GPA", 5, "y", 0.36),

    ("GEM", 0, "n", 0.48),
    ("GEM", 0, "y", 0.54),
    ("GEM", 2, "n", 0.47),
    ("GEM", 2, "y", 0.58),
    ("GEM", 5, "n", 0.53),
    ("GEM", 5, "y", 0.62),
]

MODEL_FULLNAME = {
    "G4P": "gpt-4o-audio-preview",
    "GPA": "gpt-audio",
    "GEM": "gemini-2.0-flash",
}

ORACLE_BASELINE = 1.00
TEXT_BASELINE = 0.51
RANDOM_BASELINE = 0.33


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def add_reference_lines(ax):
    ax.axhline(ORACLE_BASELINE, linestyle="-.", linewidth=1, label="Oracle (1.00)")
    ax.axhline(TEXT_BASELINE, linestyle=":", linewidth=1.5, label="Text-only (.51)")
    ax.axhline(RANDOM_BASELINE, linestyle="--", linewidth=1, label="Random (.33)")


def build_blocked_positions(rows, model_order, gap=0.9):
    """Sort rows into model blocks, return x positions + block centers."""
    rows = list(rows)
    rows.sort(key=lambda r: (model_order.index(r[0]), r[1], r[2]))  # model -> FS -> FH (n before y)

    xs, values = [], []
    meta = []  # (model, fs, fh) per bar in plotted order
    block_centers = {}

    x = 0.0
    for mi, m in enumerate(model_order):
        rows_m = [r for r in rows if r[0] == m]
        if not rows_m:
            continue

        start_x = x
        for (mm, fs, fh, inf) in rows_m:
            xs.append(x)
            values.append(inf)
            meta.append((mm, fs, fh))
            x += 1.0

        end_x = x - 1.0
        block_centers[m] = (start_x + end_x) / 2.0

        if mi != len(model_order) - 1:
            x += gap

    return xs, values, meta, block_centers


def add_model_names(ax, block_centers, model_order, fullnames, y=-0.42):
    for m in model_order:
        if m not in block_centers:
            continue
        ax.text(
            block_centers[m],
            y,
            fullnames.get(m, m),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )


def add_two_row_labels(ax, xs, meta, y_fh=-0.10, y_fs=-0.22, left_pad=0.9):
    """
    Replace x tick labels with two rows:
      FH row: n / y
      FS row: 0 / 2 / 5
    """
    # Hide default tick labels (keep tick locations for alignment)
    ax.set_xticks(xs)
    ax.set_xticklabels([""] * len(xs))

    # Row labels ("FH", "FS") a bit left of the first bar
    x_left = min(xs) - left_pad
    ax.text(x_left, y_fh, "FocusHint", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=10, clip_on=False)
    ax.text(x_left, y_fs, "FewShot", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=10, clip_on=False)

    # Per-bar labels
    for x, (m, fs, fh) in zip(xs, meta):
        ax.text(x, y_fh, fh, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, clip_on=False)
        ax.text(x, y_fs, str(fs), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, clip_on=False)


# --------------------------------------------------
# Plot
# --------------------------------------------------
model_order = ["G4P", "GPA", "GEM"]

xs, values, meta, block_centers = build_blocked_positions(
    rows=data,
    model_order=model_order,
    gap=0.9
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(xs, values, width=0.85)
add_reference_lines(ax)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Inference accuracy")

# Two-row labels (FS / FH)
add_two_row_labels(ax, xs, meta, y_fh=-0.10, y_fs=-0.22, left_pad=0.9)

# Model full names under each 6-bar block
add_model_names(ax, block_centers, model_order, MODEL_FULLNAME, y=-0.42)

ax.legend(loc="upper left", fontsize=8)

# Make room for FH row + FS row + model-name row
fig.subplots_adjust(bottom=0.48)

plt.show()
