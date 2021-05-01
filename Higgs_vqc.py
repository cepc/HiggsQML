import pandas as pd
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

data1 = pd.read_csv('signal.txt', usecols=['invariant mass of qq', 'invariant mass of vv'])
data1['type'] = 'A'
data2 = pd.read_csv('background.txt', usecols=['invariant mass of qq', 'invariant mass of vv'])
data2['type'] = 'B'
data3 = pd.concat([data1, data2])
data3 = data3.reset_index(drop=True)

train = data3.sample(frac=0.5)
test = data3[~data3.index.isin(train.index)]

colors = {'A':'red', 'B':'blue'}
train.plot.scatter(x='dijet_m', y='vis_all_rec_m', c=train.type.apply(lambda x: colors[x])) 

train_input = {'A': [], 'B': []}
test_input = {'A': [], 'B': []}
for i in train.values:
    train_input[i[2]].append([i[0], i[1]])
for i in train_input:
    train_input[i] = np.array(train_input[i])
for i in test.values:
    test_input[i[2]].append([i[0], i[1]])
for i in test_input:
    test_input[i] = np.array(test_input[i])
    
seed = 10599
aqua_globals.random_seed = seed

feature_dim=2
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
var_form = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)

vqc = VQC(optimizer, feature_map, var_form, train_input, test_input, datapoints[0])
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)
result = vqc.run(quantum_instance)

print('Prediction from datapoints set:')
print(f'  ground truth: {map_label_to_class_name(datapoints[1], vqc.label_to_class)}')
print(f'  prediction:   {result["predicted_classes"]}')

a = map_label_to_class_name(datapoints[1], vqc.label_to_class)
b = result["predicted_classes"]
r = len(a)

result = {}
for i in range(r):
    if a[i] == b[i]:
        if a[i] not in result:
            result[a[i]] = 1
        else:
            result[a[i]] += 1
a1 = (result['A']/a.count('A'))*100
b1 = (result['B']/a.count('B'))*100

print(f'The background rejection:{round(a1,3)}%.The signal efficiency:{round(b1,3)}%')
