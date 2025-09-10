import pandas as pd

# Step 1: Load the dataset
data_path = "../data/sev.csv"  # adjust path if needed
df = pd.read_csv(data_path)

# Step 2: Check the first few rows
print(df.head())

# Step 3: Drop rows with missing title or description
df = df.dropna(subset=['Description'])

# Step 4: Map 'severity' to your priority levels
severity_to_priority = {
    'blocker': 'Critical',
    'critical': 'High',
    'major': 'Medium',
    'minor': 'Low'
}

df['priority'] = df['Severity'].map(severity_to_priority)

# Step 5: Drop any rows where severity wasn't in our mapping
df = df.dropna(subset=['priority'])

# Step 6: Save the cleaned dataset
df.to_csv("../data/tickets_cleaned.csv", index=False)
print("Cleaned dataset saved to data/tickets_cleaned.csv")
print(df['priority'].value_counts())
