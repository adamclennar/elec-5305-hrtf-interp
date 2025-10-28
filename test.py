# sanity.py (quick check)
import numpy as np
d = np.load("data_npz/test/NH5_elev0.npz")


az = d["az_deg"]; freqs = d["freqs_hz"]
itd = d["itd_ms"]; L = d["magL_db"]; R = d["magR_db"]
band = (freqs >= 1000) & (freqs <= 8000)
ild = (R[:, band] - L[:, band]).mean(axis=1)

# 0) sort by azimuth [-180,180)
order = np.argsort((az+180)%360 - 180)
az, itd, ild = az[order], itd[order], ild[order]

# 1) basic sanity
print("ITD min/max (ms):", float(itd.min()), float(itd.max()))
print("ILD min/max (dB):", float(ild.min()), float(ild.max()))

# 2) should be NEGATIVE correlation (see note above)
print("corr(ITD, ILD):", float(np.corrcoef(itd, ild)[0,1]))
print("corr(ITD, -ILD):", float(np.corrcoef(itd, -ild)[0,1]))

# 3) sign-at-sides check: pick closest to +80° and -80°
def nearest_idx(target):
    return int(np.argmin(np.abs(((az-target+180)%360)-180)))
iL = nearest_idx(+80)   # ~left side
iR = nearest_idx(-80)   # ~right side
print("+80°  (left):  ITD=", float(itd[iL]), "  ILD=", float(ild[iL]), " (expect ITD>0, ILD<0)")
print("-80° (right):  ITD=", float(itd[iR]), "  ILD=", float(ild[iR]), " (expect ITD<0, ILD>0)")
