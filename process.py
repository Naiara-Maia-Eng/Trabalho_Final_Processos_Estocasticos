import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch, decimate, find_peaks
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm, probplot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from itertools import product


# ===== PRÉ-PROCESSAMENTO =====
# 1) Carrega e extrai freq_inst
mat = loadmat('LFO_phasor_data.mat')
t_full = mat['t0'].flatten()
phasor = mat['PMU1_I_2'][:,0]
mask = (t_full >= 80) & (t_full <= 210)
t = t_full[mask]
phasor = phasor[mask]

fase = np.unwrap(np.angle(phasor))
dt = t[1] - t[0]
Fs = 1.0 / dt
freq_inst = np.diff(fase) / (2*np.pi*dt)
freq_inst -= np.mean(freq_inst)
t_freq = t[:-1]

media = np.mean(freq_inst)
desvio = np.std(freq_inst)
a = freq_inst.min()
b = freq_inst.max()
x = np.linspace(a, b, 1000)   
pdf_teorica = norm.pdf(x, loc=media, scale=desvio)

print(f"Média: {media:.4f} Hz")
print(f"Desvio: {desvio:.4f} Hz")
print(f"Freq_Max: {b:.4f}")
print(f"Freq_Min: {a:.4f}")

plt.figure(figsize=(8, 4))
plt.hist(freq_inst, bins=50, density=True, edgecolor='k', alpha=0.7)
plt.xlabel('Δf (Hz)')
plt.ylabel('Densidade de frequência')
plt.plot(x, pdf_teorica, 'b-', linewidth=2, label='Curva Normal')
plt.title('Histograma da frequência instantânea')
plt.grid(True)
plt.tight_layout()
plt.show()

# Gerar o Q-Q plot (probplot)
plt.figure(figsize=(8, 5))
probplot(freq_inst, dist="norm", plot=plt)
plt.title("Q-Q Plot (probplot) - Comparação com Normal")
plt.grid(True)
plt.tight_layout()
plt.show()

plot_acf(freq_inst, lags=50)
plot_pacf(freq_inst, lags=50)
plt.show()

# 2) PSD para visualizar picos 
f_inst, Pxx_inst = welch(freq_inst, fs=Fs, nperseg=1024)
plt.semilogy(f_inst, Pxx_inst)
plt.axvline(2.0, color='r', ls='--', label='corte 2 Hz')
plt.xlim(0, 5)
plt.legend()
plt.show()

# 3) Decomposição via Butterworth 
cutoff = 2.0  # Hz, separa LFO (<2Hz) de rápida (>2Hz)
b_lp, a_lp = butter(4, cutoff/(Fs/2), btype='low')
lfo_inst = filtfilt(b_lp, a_lp, freq_inst)
fast_inst = freq_inst - lfo_inst

# 4) Plota componentes (decimado)
plt.figure(figsize=(10,4))
plt.plot(t_freq, freq_inst,    label='freq_inst decimada', alpha=0.5)
plt.plot(t_freq, lfo_inst,     label='LFO decimado (<2 Hz)')
plt.plot(t_freq, fast_inst, '--', label='Fast decimado (>2 Hz)')
plt.legend(); plt.grid(); plt.show()

plot_acf(lfo_inst, lags=50)
plot_pacf(lfo_inst, lags=50)
plt.show()

plot_acf(fast_inst, lags=50)
plot_pacf(fast_inst, lags=50)
plt.show()

# ======== MODELANDO A LFO ========
# 1.a) Decimação dos dados
decim_factor = 10
lfo_ds = decimate(lfo_inst, decim_factor, ftype='iir', zero_phase=True)
Fs_ds = Fs / decim_factor
t_ds = t_freq[::decim_factor]

print(f"Decimation factor: {decim_factor}, New Fs: {Fs_ds:.2f} Hz, Length: {len(lfo_ds)}")

plot_acf(lfo_ds, lags=50)
plot_pacf(lfo_ds, lags=50)
plt.show()


x_raw = lfo_ds
fs = Fs_ds

# 1) Pico dominante da LFO -> período sazonal s

f2, Pxx2 = welch(lfo_ds, fs=Fs_ds, nperseg=1024, nfft=2048)
peaks, _ = find_peaks(Pxx2, distance=int(0.5/(f2[1]-f2[0])))
mask_peaks = (f2[peaks] >= 0.2) & (f2[peaks] <= 5.0)
peaks = peaks[mask_peaks]
top2 = peaks[np.argsort(Pxx2[peaks])[::-1][:2]]
modo1, modo2 = f2[top2]
period1 = int(round(Fs_ds/modo1))
period2 = int(round(Fs_ds/modo2))
print(f"Modos detectados (decimado): {modo1:.3f} Hz (T={period1}), {modo2:.3f} Hz (T={period2})")

f0 = 0.85  # Hz
s = int(round(Fs_ds/f0))
print(f"LFO fixada: f0={f0} Hz -> s={s} amostras")


x = (x_raw - np.mean(x_raw)) / np.std(x_raw, ddof=1)

n = np.arange(len(x))
w = 2*np.pi*f0/fs         # radianos por amostra

X = np.column_stack([np.sin(w*n), np.cos(w*n), np.sin(2*w*n), np.cos(2*w*n)])

cands = []
for p, q in product(range(0,5), range(0,5)):   
    try:
        m = SARIMAX(
            x, exog=X,
            order=(p, 0, q),                 # ARMA(p,q) no resíduo
            seasonal_order=(0,0,0,0),
            enforce_stationarity=True,
            enforce_invertibility=True
        ).fit(disp=False)
        # diagnóstico rápido
        lb = acorr_ljungbox(m.resid, lags=[10, 20, s, 2*s], return_df=True)
        minp = float(lb['lb_pvalue'].min())
        cands.append((minp, m.aic, (p,q), m))
        print(f"ARMA({p},{q}) com harmônicos: AIC={m.aic:.1f}, min p(LB)={minp:.3g}")
    except Exception as e:
        pass

# escolhe priorizando resíduos "brancos" (maior min p), depois menor AIC
cands.sort(key=lambda z: (-z[0], z[1]))
minp, best_aic, (p,q), best = cands[0]
print(f"\n>> Escolhido: Harmônicos @ f0 + ARMA({p},{q}) | AIC={best_aic:.1f} | min p(LB)={minp:.3g}")

res = best.resid
plot_acf(res, lags=50); plt.title("ACF resíduos (harmônicos + ARMA)"); plt.show()
plot_pacf(res, lags=50); plt.title("PACF resíduos (harmônicos + ARMA)"); plt.show()
print(acorr_ljungbox(res, lags=[10,20,s,2*s], return_df=True))

print(best.summary())

# (re)calcula média e desvio da LFO decimada (x_raw) para desfazer a padronização
mu  = np.mean(x_raw)
sig = np.std(x_raw, ddof=1)

# ====== RECONSTRUÇÃO IN-SAMPLE =======
pred_in = best.get_prediction(exog=X)
yhat_in_std = np.asarray(pred_in.predicted_mean)
yhat_in     = yhat_in_std * sig + mu

# IC (~95%) in-sample em escala original 
ci_in = pred_in.conf_int()
if isinstance(ci_in, np.ndarray):
    ci_in_lo = ci_in[:, 0] * sig + mu
    ci_in_hi = ci_in[:, 1] * sig + mu
else:
    ci_in_lo = ci_in.iloc[:, 0].to_numpy() * sig + mu
    ci_in_hi = ci_in.iloc[:, 1].to_numpy() * sig + mu

# métricas de reconstrução
mae_in = mean_absolute_error(x_raw, yhat_in)
rmse_in = np.sqrt(mean_squared_error(x_raw, yhat_in))
print(f"[Harmônicos+ARMA] Reconstrução in-sample: MAE={mae_in:.6f}  RMSE={rmse_in:.6f}")


# plot reconstrução
plt.figure(figsize=(10,4))
plt.plot(t_ds, x_raw, label='LFO (observada)', alpha=0.65)
plt.plot(t_ds, yhat_in, label='Reconstrução (harmônicos+ARMA)', lw=2)
plt.fill_between(t_ds, ci_in_lo, ci_in_hi, alpha=0.15, label='IC ~95%')
plt.title('Reconstrução in-sample – LFO (harmônicos + ARMA)')
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# ===== FORECAST ========

lookback = 250
steps    = 120

nobs  = len(x)
start = max(0, nobs - lookback)
end   = nobs + steps - 1

# --- exógenas FUTURAS (fora da amostra) ---
n_fut = np.arange(nobs, end + 1)
X_fut = np.column_stack([
    np.sin(w*n_fut),  np.cos(w*n_fut),
    np.sin(2*w*n_fut), np.cos(2*w*n_fut)
])

# --- previsão no espaço padronizado ---
pred = best.get_prediction(start=start, end=end, dynamic=False, exog=X_fut)
yhat_std = np.asarray(pred.predicted_mean)
ci_std   = np.asarray(pred.conf_int(alpha=0.05))  # (N,2)

# --- DESPADRONIZA (volta à escala do x_raw) ---
yhat = yhat_std * sig + mu
ci_lo = ci_std[:,0] * sig + mu
ci_hi = ci_std[:,1] * sig + mu

# eixos de tempo
t_hist = t_ds[start:nobs]
t_fut  = t_ds[-1] + (np.arange(1, steps+1) / Fs_ds)
t_all  = np.concatenate([t_hist, t_fut])

# separa parte dentro/fora da amostra
k = len(t_hist)
yhat_in,  yhat_fu  = yhat[:k],  yhat[k:]
ci_in_lo, ci_fu_lo = ci_lo[:k], ci_lo[k:]
ci_in_hi, ci_fu_hi = ci_hi[:k], ci_hi[k:]

# ===== PLOT no MESMO EIXO/ESCALA do sinal original =====
plt.figure(figsize=(12,3))
plt.plot(t_ds, x_raw, color='steelblue', label='Histórico (x_raw)')
plt.plot(t_hist, yhat_in, color='orange', lw=2, label='Forecast (in+out)')
plt.fill_between(t_hist, ci_in_lo, ci_in_hi, color='gray', alpha=0.25, label='IC 95%')
plt.plot(t_fut, yhat_fu, color='orange', lw=2)
plt.fill_between(t_fut, ci_fu_lo, ci_fu_hi, color='gray', alpha=0.25)
plt.axvline(t_ds[-1], color='k', ls='--', lw=1, label='Início forecast')
plt.title('Forecast harmônicos + ARMA com IC 95% (na escala do sinal)')
plt.xlabel('tempo'); plt.ylabel('LFO (decimada)')
plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()



