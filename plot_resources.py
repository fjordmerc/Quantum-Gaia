
"""
Generate Figure F7: Resource Usage.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Use expanduser to properly handle the ~ path
df = pd.read_csv(os.path.expanduser('~/env_QGA/results/resource_usage.csv'))
df.set_index('component')[['qubits','gate_count']].plot.bar(rot=0)
plt.ylabel('Count')
plt.title('Resource Usage Comparison')
plt.savefig(os.path.expanduser('~/env_QGA/results/plots/F7_resource_usage.pdf'), dpi=300)
plt.show()  # Optional: to display the plot
