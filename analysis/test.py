# analysis scratch
import numpy as np
d = np.load('data_npz/test/hrtf_M_hrtf_B_elev0.npz')
az, itd = d['az_deg'], d['itd_ms']
bad = np.where(np.abs(itd) > 0.2)[0]   # tune threshold
print(list(zip(az[bad], itd[bad]))[:10])
