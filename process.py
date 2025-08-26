import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch, decimate, find_peaks
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import yule_walker
from scipy.signal import lfilter
from statsmodels.tsa.stattools import acf
from scipy.stats import norm, probplot
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
x = np.linspace(a, b, 1000)   # 1000 pontos uniformes entre a e b
pdf_teorica = norm.pdf(x, loc=media, scale=desvio)

print(f"Média: {media:.4f} Hz")
print(f"Desvio: {desvio:.4f} Hz")
print(f"Freq_Max: {a:.4f}")
print(f"Freq_Min: {b:.4f}")

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


# 1.a) Decimação dos dados
decim_factor = 10
freq_ds = decimate(freq_inst, decim_factor, ftype='iir', zero_phase=True)
Fs_ds = Fs / decim_factor
t_ds = t_freq[::decim_factor]

print(f"Decimation factor: {decim_factor}, New Fs: {Fs_ds:.2f} Hz, Length: {len(freq_ds)}")

# 2) PSD para visualizar picos (decimado)
f_ds, Pxx_ds = welch(freq_ds, fs=Fs_ds, nperseg=1024)
plt.semilogy(f_ds, Pxx_ds)
plt.axvline(1.0, color='r', ls='--', label='corte 1 Hz')
plt.xlim(0, 5)
plt.legend()
plt.show()

# 3) Decomposição via Butterworth (aplicada ao decimado)
cutoff = 1.0  # Hz, separa LFO (<1Hz) de rápida (>1Hz)
b_lp, a_lp = butter(4, cutoff/(Fs_ds/2), btype='low')
lfo_ds = filtfilt(b_lp, a_lp, freq_ds)
fast_ds = freq_ds - lfo_ds

# 4) Plota componentes (decimado)
plt.figure(figsize=(10,4))
plt.plot(t_ds, freq_ds,    label='freq_inst decimada', alpha=0.5)
plt.plot(t_ds, lfo_ds,     label='LFO decimado (<1 Hz)')
plt.plot(t_ds, fast_ds, '--', label='Fast decimado (>1 Hz)')
plt.legend(); plt.grid(); plt.show()

# 5) Detecta modos no PSD do decimado
f2, Pxx2 = welch(fast_ds, fs=Fs_ds, nperseg=1024, nfft=2048)
peaks, _ = find_peaks(Pxx2, distance=int(0.5/(f2[1]-f2[0])))
mask_peaks = (f2[peaks] >= 0.2) & (f2[peaks] <= 5.0)
peaks = peaks[mask_peaks]
top2 = peaks[np.argsort(Pxx2[peaks])[::-1][:2]]
modo1, modo2 = f2[top2]
period1 = int(round(Fs_ds/modo1))
period2 = int(round(Fs_ds/modo2))
print(f"Modos detectados (decimado): {modo1:.3f} Hz (T={period1}), {modo2:.3f} Hz (T={period2})")

# 6) ADF e ARIMA na LFO decimada
# Opção 1: unpack com star
adf_stat, p_value, *_ = adfuller(lfo_ds)

plot_acf(lfo_ds, lags=50)
plot_pacf(lfo_ds, lags=50)
plt.show()

plot_acf(fast_ds, lags=50)
plot_pacf(fast_ds, lags=50)
plt.show()

print(f"LFO decimado ADF: {adf_stat:.3f}, p={p_value:.3f}")
d = 1 if p_value >= 0.05 else 0
arima_lfo = ARIMA(lfo_ds, order=(2,d,1)).fit()
print("LFO ARIMA LB-p:", acorr_ljungbox(arima_lfo.resid, lags=[10]).iloc[0,1])

resid_lfo = arima_lfo.resid

plot_acf(arima_lfo.resid, lags=50)
plot_pacf(arima_lfo.resid, lags=50)
plt.show()

# 7) Pré-whitening AR(p) na fast decimada
p = 20
rho, sigma = yule_walker(fast_ds, order=p, method='mle')
ar = np.r_[1, -rho]
resid_ar = lfilter(ar, [1], fast_ds)
p_lb_ar = acorr_ljungbox(resid_ar, lags=[10]).iloc[0,1]
print(f"Fast decimado AR({p}) LB-p on resid: {p_lb_ar:.4f}")

resid = resid_ar

# calcula a função de autocorrelação
acf_vals = acf(resid, nlags=10, fft=True)
max_acf = np.max(np.abs(acf_vals[1:]))   # ignora lag-0
print(f"Máxima ACF(resíduo) em lags 1–10: {max_acf:.4f}")

# plota o correlograma
plot_acf(resid, lags=10)
plt.title("ACF dos resíduos (decimado)")
plt.show()

# 8) Forecast do componente LFO (decimado)
# Use o modelo ARIMA já ajustado (arima_lfo)
n_forecast = 1000
lfo_forecast = arima_lfo.forecast(steps=n_forecast)
t_forecast = np.arange(t_ds[-1] + dt, t_ds[-1] + dt * (n_forecast + 1), dt)

plt.figure(figsize=(10, 4))
plt.plot(t_ds, lfo_ds, label='LFO (histórico)')
plt.plot(t_forecast, lfo_forecast, 'r--', label='LFO (previsão)')
plt.legend()
plt.title('Previsão do Componente LFO')
plt.grid()
plt.show()

# Gerar o Q-Q plot (probplot)
plt.figure(figsize=(8, 5))
probplot(resid_lfo, dist="norm", plot=plt)
plt.title("Q-Q Plot (probplot) - Comparação com Normal")
plt.grid(True)
plt.tight_layout()
plt.show()

# Simulação direta usando o método simulate do próprio modelo
np.random.seed(1)

# Utilize as últimas observações como ponto de referência inicial (anchor='end')
y_sintetico = arima_lfo.simulate(len(lfo_ds), anchor='end', measurement_shocks=np.random.normal(0, sigma/5, len(lfo_ds)))


# Para manter escala original, some o valor médio inicial
y_sintetico += np.mean(lfo_ds)

# Visualização definitiva
plt.figure(figsize=(10,4))
plt.plot(t_ds, lfo_ds, label='LFO Original', linewidth=2)
plt.plot(t_ds, y_sintetico, '--', label='LFO Sintético (simulate direto)', alpha=0.8)
plt.legend()
plt.title("Comparação definitiva LFO original vs. sintético")
plt.grid()
plt.tight_layout()
plt.show()

y_sint = y_sintetico
N = len(y_sint)
n_test = int(np.ceil(0.2 * N))
n_train = N - n_test

y_train_sint, y_test_sint = y_sint[:n_train], y_sint[n_train:]

model_sint = ARIMA(y_train_sint, order=(2,1,1)).fit()

fc_sint = model_sint.get_forecast(steps=n_test)
y_pred_sint = fc_sint.predicted_mean

mae_sint = mean_absolute_error(y_test_sint, y_pred_sint)
rmse_sint = np.sqrt(mean_squared_error(y_test_sint, y_pred_sint))

print(f"MAE Sintético:  {mae_sint:.4f}")
print(f"RMSE Sintético: {rmse_sint:.4f}")


