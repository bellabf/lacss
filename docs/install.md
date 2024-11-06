#### Linux with NVidia GPU
```
# check nvidia driver version >= 525.60.13
cat /proc/driver/nvidia/version

# check python version is 3.10 or higher
python --verison

# install with cuda support
pip install lacss[cuda12]

# alternatively, if you want to train your own model
pip install lacss[train]
```

#### Linux wih CPU
```
pip install lacss
```

#### Linux / Mac / Windows / Other Configurations
- install Jax following the [official guide](https://jax.readthedocs.io/en/latest/installation.html).
- `pip install lacss`

