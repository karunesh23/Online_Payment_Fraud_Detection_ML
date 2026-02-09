import pandas as pd

# Load big PaySim dataset
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Take random sample
df_small = df.sample(n=100000, random_state=42)

# Save smaller dataset
df_small.to_csv("paysim_small.csv", index=False)

print("✅ paysim_small.csv created")
print("Rows:", df_small.shape[0])
