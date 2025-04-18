import time
from utils import check_and_download_beth_data

start = time.time()
DROPBOX_URL = "https://www.dropbox.com/scl/fi/dgn7n5i6qe14cdwciomy6/beth_dataset.zip?rlkey=8pbjsgewyz68m3mtwfj5a92d8&st=rvx25oiq&dl=1"
csv_files = check_and_download_beth_data(DROPBOX_URL)
end = time.time()
print(f"total time took to download data from server (if needed) is: {end - start}")