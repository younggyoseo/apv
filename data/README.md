# Raw Data

We provide raw data we used in our experiments on Meta-world(`metaworld_raw_data.pkl`) and DeepMind Control Suite (`dmc_raw_data.pkl`). To compute IQM, you can utilize [rliable](https://github.com/google-research/rliable) repository.

## How to load the data
To load the raw data we used in our experiments:
```
import pickle
with open('metaworld_raw_data.pkl', 'rb') as f:
    metaworld_raw_data = pickle.load(f)
```

`metaworld_raw_data` is a dictionary that consists like this:
```
metaworld_raw_data = {
    'DreamerV2': {
        'metaworld_lever_pull': np.array([8, 25]),
        ...,
        'metaworld_reach': np.array([8, 25]),
    },
    'APV': {
        'metaworld_lever_pull': np.array([8, 25]),
        ...,
        'metaworld_reach': np.array([8, 25]),
    }
}
```

## Evaluation Protocols
- Meta-world Experiments: we report the average success rate over 10 trials at `[5000, 15000, ..., 250000]` environment steps (with action repeat of 1).

- DeepMind Control Suite Experiments: we report the episode return over 1 episode at `[2000, 22000, ..., 982000]` environment steps (with aciton repeat of 2)