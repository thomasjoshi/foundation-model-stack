import matplotlib.pyplot as plt
import pandas as pd
import sys

# Usage: python scripts/plot_attention_benchmarks.py runtime.csv memory.csv
if len(sys.argv) != 3:
    print("Usage: python scripts/plot_attention_benchmarks.py <runtime_csv> <memory_csv>")
    sys.exit(1)

runtime_csv = sys.argv[1]
memory_csv = sys.argv[2]

# Load data
df_runtime = pd.read_csv(runtime_csv)
df_memory = pd.read_csv(memory_csv)

# Plot runtime vs sequence length
plt.figure(figsize=(8, 5))
for paged, label, style in zip([False, True], ["Non-Paged Attention", "Paged Attention"], ["-", "--"]):
    df = df_runtime[df_runtime["paged"] == paged]
    plt.plot(df["seq_len"], df["runtime_ms"], style, label=label, marker="o")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Sequence Length")
plt.ylabel("Runtime (ms) [Fwd + Bwd]")
plt.title("Attention Runtime (Fwd Pass + Bwd Pass)")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("attention_runtime_vs_seqlen.png")

# Plot memory vs sequence length
plt.figure(figsize=(8, 5))
for paged, label, style in zip([False, True], ["Non-Paged Attention", "Paged Attention"], ["-", "--"]):
    df = df_memory[df_memory["paged"] == paged]
    plt.plot(df["seq_len"], df["memory_gb"], style, label=label, marker="o")
plt.xscale("log")
plt.xlabel("Sequence Length")
plt.ylabel("Memory Footprint (GB)")
plt.title("Attention Memory Usage")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("attention_memory_vs_seqlen.png")

print("Saved plots: attention_runtime_vs_seqlen.png, attention_memory_vs_seqlen.png") 