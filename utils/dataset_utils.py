# Dataset classes for Franka demonstration data and scene encoder training data
# Most functions follow the original Diffusion Policy implementation

import numpy as np
import torch
import h5py
import pickle
import time

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    denom = stats['max'] - stats['min']
    denom[denom == 0] = 1.0  
    # nomalize to [0,1]
    ndata = (data - stats['min']) / denom
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    denom = stats['max'] - stats['min']
    denom[denom == 0] = 1.0  

    ndata = (ndata + 1) / 2
    data = ndata * denom + stats['min']
    return data

# franka dataset class for loading data from an HDF5 file
class FrankaStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_paths, pred_horizon, obs_horizon, action_horizon, stats = None):
        # structure initialization
        train_data = {
            'action': [],
            'obs': [],
            'dones': [],
        }

        for dataset_path in dataset_paths:
            with h5py.File(dataset_path, 'r') as f:
                demos = list(f['data'].keys())
                
                # loading every demonstration data from this file
                for demo in demos:
                    demo_group = f['data'][demo]
                    
                    train_data['action'].append(demo_group['actions'][:])
                    object1 = demo_group['obs/object'][:, :3]
                    object2 = demo_group['obs/object'][:, 3:6]

                    combined_obs = np.concatenate([
                        demo_group['states'][:],
                        object1,          
                        object2,
                    ], axis=-1)

                    train_data['obs'].append(combined_obs)
                    train_data['dones'].append(demo_group['dones'][:])

        for key in train_data:
            if len(train_data[key]) == 0:
                raise ValueError(f"La clé {key} a une liste vide, ce qui empêche la concaténation.")
            train_data[key] = np.concatenate(train_data[key], axis=0)

        episode_ends = np.where(train_data['dones'] == 1)[0]

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1
        )

        if stats == None:
            stats = {
                'action': get_data_stats(train_data['action']),
                'robot_state': get_data_stats(train_data['obs'][:, :7]),
            }

            stats_file_name = f"stats/stats_{time.strftime('%Y-%m-%d_%H-%M')}.pkl"
            with open(stats_file_name, "wb") as f:
                pickle.dump({'action': stats['action'], 'robot_state': stats['robot_state']}, f)
            print(f"Normalizing stats saved in {stats_file_name}")

        normalized_obs = np.concatenate([
            normalize_data(train_data['obs'][:, :7], stats['robot_state']),
            train_data['obs'][:, 7:],  # objetcs' coordinates are not normalized
        ], axis=-1)

        normalized_train_data = {
            'action': normalize_data(train_data['action'], stats['action']),
            'obs': normalized_obs,
            'dones': train_data['dones'],
        }
            
        # save parameters
        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # obtain start and end indices
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # obtain normalized data
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]

        return nsample

class RandomSceneDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # generate a goal
        goal = torch.empty(3).uniform_(0.3, 0.7)
        goal[1] = torch.empty(1).uniform_(-0.3, 0.3)
        goal[2] = 0.1  # fixed z 

        # generate n objects
        objects = []
        for _ in range(4):
            obj = torch.empty(3).uniform_(0.3, 0.7)
            obj[1] = torch.empty(1).uniform_(-0.3, 0.3)
            obj[2] = 0.1
            objects.append(obj)
        objects = torch.stack(objects)  # (4, 3)

        # nobs[:, -1, 7:10] => goal
        # nobs[:, -1, 10:13], [13:16], ... => objects

        nobs = torch.zeros(2, 25)  # obs_horizon=2, dim=25 
        nobs[-1, 7:10] = goal
        for i, obj in enumerate(objects):
            nobs[-1, 10 + i*3: 13 + i*3] = obj

        return {
            'obs': nobs  # (1, T=2, D=25)
        }