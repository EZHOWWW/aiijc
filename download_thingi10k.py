import thingi10k
import numpy as np
import warnings

# Suppress a specific warning from the library if it appears
warnings.filterwarnings("ignore", category=UserWarning, module="thingi10k")

# Initialize the library to ensure data is available
# This will download and cache the dataset metadata
thingi10k.init()

print("Starting to process dataset...")
# Loop through all entries in the dataset
for entry in thingi10k.dataset():
    file_id = entry["file_id"]
    author = entry.get("author", "N/A")  # Use .get() for safety

    # --- FIX STARTS HERE ---
    # Instead of calling the problematic function, we load the file manually.
    # try:
    # entry["file_path"] points to a compressed NumPy file (.npz)
    with np.load(entry["file_path"]) as data:
        # The core of the fix: use a concrete dtype like np.float64
        # instead of the old np.floating.
        print(dict(data))
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        # 'faces' is the correct key for facets in their .npz files
        facets = np.asarray(data["facets"], dtype=np.int32)
        break

    # Do something with the loaded data
    print(
        f"Successfully loaded: ID={file_id}, Author={author}, Vertices={vertices.shape}, Facets={facets.shape}"
    )

    # except Exception as e:
    #     # This makes your script robust: if a file is broken, it will print an error and continue.
    #     print(f"Could not process file_id {file_id}. Error: {e}")
    #     continue
    # --- FIX ENDS HERE ---

print("\nProcessing complete.")

# You can still use help() to get more info about the library structure
# help(thingi10k)
