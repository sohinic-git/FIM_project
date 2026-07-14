import pandas as pd 
import numpy as np
import scipy as stats



np.random.seed(42)
np.set_printoptions(suppress=True)


##Generate Baseline logs

baseline_hits = np.random.randint(2, 20, size = 40)

payloads_test = ((baseline_hits * np.random.randint(10, size=40)) +np.random.gamma(shape = 2, scale = 1, size = 40))

baseline_df = pd.DataFrame( {
    'API_hits' : baseline_hits,
    'payloads_test' : payloads_test 

})

cov_matrix = baseline_df.cov()

clean_fim_matrix = np.linalg.inv(cov_matrix)



#Generate Attacker logs


attacker_hits = np.random.randint(2, 10, size = 10)

rng = np.random.default_rng()

payloads_test_attacker = ((28 * rng.random(size=10)))
attcker_df = pd.DataFrame( {
    'API_hits' : attacker_hits,
    'payloads_test' : payloads_test_attacker 

})

##Evasion

compromised_df = pd.concat( [baseline_df[0:30], attcker_df],ignore_index=True)

compromised_df = compromised_df.sample(frac=1).reset_index(drop=True)


cov_matrix_attacker = compromised_df.cov()

attack_fim_matrix = np.linalg.inv(cov_matrix_attacker)

## Statistical approaches to detect anomalous activity

###1. Mahalanobis approach

##Baseline  calculation:

delta = baseline_df.iloc[26] - baseline_df.mean()
x_mu_values = delta.to_numpy()

baseline_cov = cov_matrix.to_numpy()

inv_baseline_cov = np.linalg.inv(baseline_cov)

mahal = x_mu_values @ inv_baseline_cov @x_mu_values
print(f'Baseline_distance: {np.sqrt(mahal)}')

##Compromised calculation:

delta_a = compromised_df.iloc[26] - baseline_df.mean()
x_mu_values_a = delta_a.to_numpy()

attck_cov = cov_matrix_attacker.to_numpy()

inv_attack_cov = np.linalg.inv(attck_cov)

mahal_a = x_mu_values_a @ inv_attack_cov @ x_mu_values_a
print(f'Anomaly_distance: {np.sqrt(mahal_a)}')

###2. FIM approach

print("Clean Fim Matrix: ")
print(f'{clean_fim_matrix}')

print("Anomalous FIM matrix: ")
print(f'{attack_fim_matrix}')


print("Clean Fim Matrix Determinant: ")

clean_det = np.linalg.det(clean_fim_matrix)

print(clean_det)

print("Anomalous Fim Matrix Determinant: ")

anon_det = np.linalg.det(attack_fim_matrix)
print(anon_det)

scale = 100000
scaled_det_c = clean_det * scale
scaled_det_anon = anon_det * scale


info_change = (((scaled_det_anon - scaled_det_c)/scaled_det_c) * 100)

print(f'Information Vol change: {info_change:.2f}%')

