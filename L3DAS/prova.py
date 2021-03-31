import pickle
import numpy as np
import os




l = []
target = []
for i in range(20):
    n = np.random.randint(20000) + 2000
    n = 16000 * 10
    sig = np.random.sample(n)
    target.append(sig)
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

print (data[0].shape)
