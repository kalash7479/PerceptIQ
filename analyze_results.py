"""
analyze_results.py

Generates an analytics summary from PerceptIQ log (./output/detection_log.csv).

Usage:
    python analyze_results.py

Requires:
    pip install pandas seaborn
    (matplotlib should already be installed from the main project setup)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH = "./output/detection_log.csv"
OUT_DIR = "./output"


def main():
    if not os.path.exists(LOG_PATH):
        print(f"[ERROR] No detection log found at {LOG_PATH}. Run app.py first.")
        return

    df = pd.read_csv(LOG_PATH)
    print(
        f"Loaded {len(df)} detection rows from {df['frame'].nunique()} frames "
        f"and {df['track_id'].nunique()} unique tracked vehicles.\n"
    )

    # --- Summary stats ---
    print("Risk level counts:")
    print(df["risk_level"].value_counts(), "\n")

    closest = df.loc[df["distance_m"].idxmin()]
    print(
        f"Closest approach: {closest['vehicle_type']} at {closest['distance_m']}m "
        f"(frame {closest['frame']}, t={closest['timestamp_sec']}s, "
        f"risk={closest['risk_level']})\n"
    )

    danger_events = df[df["risk_level"] == "DANGER"]
    print(
        f"DANGER-level events: {len(danger_events)} detections across "
        f"{danger_events['frame'].nunique()} frames\n"
    )

    # --- Plot 1: Risk level distribution ---
    plt.figure(figsize=(6, 4))
    order = ["SAFE", "CAUTION", "WARNING", "DANGER"]
    palette = {
        "SAFE": "#2ecc71",
        "CAUTION": "#f1c40f",
        "WARNING": "#e67e22",
        "DANGER": "#e74c3c",
    }
    sns.countplot(data=df, x="risk_level", order=order, palette=palette)
    plt.title("Detection Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Detections")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/risk_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/risk_distribution.png")

    # --- Plot 2: Closest vehicle distance over time ---
    closest_per_frame = (
        df.groupby("frame")
        .agg(timestamp_sec=("timestamp_sec", "first"), min_distance=("distance_m", "min"))
        .reset_index()
    )

    plt.figure(figsize=(9, 4))
    plt.plot(
        closest_per_frame["timestamp_sec"],
        closest_per_frame["min_distance"],
        color="#3498db",
        linewidth=1.2,
    )
    plt.axhline(8, color="#e74c3c", linestyle="--", linewidth=1, label="Danger threshold (8m)")
    plt.title("Closest Vehicle Distance Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/distance_timeline.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/distance_timeline.png")

    # --- Plot 3: Tracked vehicle count per frame ---
    count_per_frame = (
        df.groupby("frame")["track_id"].nunique().reset_index(name="vehicle_count")
    )
    plt.figure(figsize=(9, 4))
    plt.plot(count_per_frame["frame"], count_per_frame["vehicle_count"], color="#9b59b6")
    plt.title("Tracked Vehicle Count Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Vehicle Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vehicle_count_timeline.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR}/vehicle_count_timeline.png")


if __name__ == "__main__":
    main()
