import pickle
import numpy as np
import os




l = []
target = []
for i in range(4):
    n = 160000
    n_target = 160000
    sig = np.random.sample(n)
    sig_target = np.random.sample(n_target).reshape((1, n_target))
    target.append(sig_target)
    sig = np.vstack((sig,sig,sig,sig))
    l.append(sig)



output_path = '../prova_pickle'
if not os.path.isdir(output_path):
    os.mkdir(output_path)

with open(os.path.join(output_path,'training_predictors.pkl'), 'wb') as f:
    pickle.dump(l, f)
with open(os.path.join(output_path,'training_target.pkl'), 'wb') as f:
    pickle.dump(target, f)
with open(os.path.join(output_path,'validation_predictors.pkl'), 'wb') as f:
    pickle.dump(l, f)
with open(os.path.join(output_path,'validation_target.pkl'), 'wb') as f:
    pickle.dump(target, f)
with open(os.path.join(output_path,'test_predictors.pkl'), 'wb') as f:
    pickle.dump(l, f)
with open(os.path.join(output_path,'test_target.pkl'), 'wb') as f:
    pickle.dump(target, f)
'''
np.save(os.path.join(output_path,'training_predictors.npy'), l)
np.save(os.path.join(output_path,'training_target.npy'), l)
np.save(os.path.join(output_path,'validation_predictors.npy'), l)
np.save(os.path.join(output_path,'validation_target.npy'), l)
np.save(os.path.join(output_path,'test_predictors.npy'), l)
np.save(os.path.join(output_path,'test_target.npy'), l)
'''

with open(os.path.join(output_path,'training_predictors.pkl'), 'rb') as f:
    data = pickle.load(f)
with open(os.path.join(output_path,'training_target.pkl'), 'rb') as f:
    data2 = pickle.load(f)

print (data[0].shape)
print (data2[0].shape)
