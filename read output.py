import numpy as np
output = np.load('./output/nsd_ensemble/repeat100/size300/S2_V1v.npy', allow_pickle=True).tolist()
print('Mean accuracy:', np.nanmean(output['le_acc']))
print('Std accuracy:', np.nanstd(output['le_acc']))