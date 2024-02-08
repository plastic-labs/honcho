import os
import re

# Open the source file
this_dir = os.path.dirname(os.path.abspath(__file__))
source_file_path = os.path.join(this_dir, "../sdk/honcho/client.py")
with open(source_file_path, "r") as source_file:
    source_code = source_file.read()

# Use regex to remove async mentions
sync_code = re.sub(r"async\s", "", source_code)
sync_code = re.sub(r"await\s", "", sync_code)
sync_code = re.sub(r"Async", "", sync_code)

# Write the modified code to the destination file
destination_file_path = os.path.join(this_dir, "../sdk/honcho/sync_client.py")
with open(destination_file_path, "w") as destination_file:
    destination_file.write(sync_code)
