# calibration weights and tau estimation

# value from 2019 with a buffer
calibration_target = 0.032043

a = result_logodds.params[0]
b = result_logodds.params[1]
sc = np.array(app_frame['agg_score_logodds'])
cal_tar = calibration_target

def to_minimize(T, A, B, scores, N, target):
  return (1 / N * sum(1 / (1 + np.exp(-(T + A + B * scores)))) - target) ** 2

res = minimize(to_minimize, x0 = 1, args=(a, b, sc, len(sc), cal_tar),  method='Nelder-Mead',
               options={'xtol': 1e-10, 'disp': True})

# Tau
tau = res.x[0]
print(cal_tar)
print(tau)
to_minimize(tau, a, b, sc, len(sc), cal_tar)