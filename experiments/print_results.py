import torch, numpy as np

lines = []
r = torch.load('inspection_results/label_efficiency_results.pt', weights_only=False)
fracs = sorted(r.keys())
lines.append('=== UCI HAR LABEL EFFICIENCY ===')
lines.append(f"{'Frac':<8}  {'Scratch (mean +/- std)':<25}  {'Probe (mean +/- std)':<25}  {'Fine-Tune (mean +/- std)':<25}")
for f in fracs:
    s = r[f]['scratch']; p = r[f]['probe']; ft = r[f]['finetune']
    lines.append(f"{f*100:.0f}%       {np.mean(s):.4f} +/- {np.std(s):.4f}          {np.mean(p):.4f} +/- {np.std(p):.4f}          {np.mean(ft):.4f} +/- {np.std(ft):.4f}")

r2 = torch.load('inspection_results/sisfall_transfer_results.pt', weights_only=False)
lines.append('')
lines.append('=== SISFALL TRANSFER ===')
lines.append(f"{'Metric':<12}  {'Scratch (mean +/- std)':<25}  {'MAE Transfer (mean +/- std)':<25}")
for k in ['f1','recall','precision']:
    s = r2['scratch'][k]; ft = r2['finetune'][k]
    lines.append(f"{k:<12}  {np.mean(s):.4f} +/- {np.std(s):.4f}          {np.mean(ft):.4f} +/- {np.std(ft):.4f}")

output = '\n'.join(lines)
print(output)
with open('inspection_results/final_results_summary.txt', 'w') as f:
    f.write(output + '\n')
print('\nSaved to inspection_results/final_results_summary.txt')
