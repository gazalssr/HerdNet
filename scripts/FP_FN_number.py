import pandas as pd

# Load the data from a CSV file
df = pd.read_csv('/herdnet/test_output/new_Test_Final_detection_file.csv')

# Calculate FPs and FNs
fp = df[(df['binary'] == 1) & (df['Ground_truth'] == 0)]
fn = df[(df['binary'] == 0) & (df['Ground_truth'] == 1)]

# Extract the image IDs for FPs and FNs
fp_image_ids = fp['images'].unique()
fn_image_ids = fn['images'].unique()

# Display results
print("False Positives (FP) image IDs:")
print(fp_image_ids)

print("\nFalse Negatives (FN) image IDs:")
print(fn_image_ids)

# Count the number of false negatives
fp_count = len(fp_image_ids)
print("False Negatives (FN) count:", fp_count)
# Count the number of false negatives
fn_count = len(fn_image_ids)
print("False Negatives (FN) count:", fn_count)

# Total count of false positives and false negatives
total_fp_fn_count = fp_count + fn_count
print("Total False Positives (FP) and False Negatives (FN) count:", total_fp_fn_count)