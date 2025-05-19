# %%
import numpy as np
import inharmonicon as inharm
#from progressbar import Bar
import tarfile
import os.path


# %%

def make_tarfile(output_filename, source_dir):
    print('Compressing to tar...')
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print('Done.')
# %%
    
# stimulus length in seconds
stim_length = .05

# jitter rate for inharmonic sounds
jr = 0.1

# lowest frequency
fmin = 10

# maximal frequency (can be over-the-top - it's going to be capped at Nyquist either way)
fmax = 100

# filter specification
filter_spec = {
    'freq': 3000,
    'order': 1,
    'type': 'lowpass'
}

# how many inharmonic sounds in pool?
# results in 3 * no_sounds, every jitter pattern is used for 3 sounds (std, pos, neg)
no_sounds = 20 

# fundamental frequencies for standard and deviants
f0_std = 300
f0_pos = 350


# filepath for harmonic sounds
fpath = 'soundpool/'  

# Create directories if they don't exist
os.makedirs(fpath, exist_ok = True)

# make harmonic series and fully harmonic sounds
harmonic_series = inharm.Harmonics(f0 = f0_std, fmin = fmin, fmax = fmax)
harmonic_sound = inharm.Sound(harmonic_series, length = stim_length)
harmonic_sound.filter(**filter_spec)
harmonic_sound.save(f"{fpath}{f0_std}_0.wav")

harmonic_series = inharm.Harmonics(f0 = f0_pos, fmin = fmin, fmax = fmax)
harmonic_sound = inharm.Sound(harmonic_series, length = stim_length)
harmonic_sound.filter(**filter_spec)
harmonic_sound.save(f"{fpath}{f0_pos}_0.wav")

omission = inharm.Sound(None, length = stim_length)
omission.save(f"{fpath}omi.wav")
# %%

for i in range(1,no_sounds):
    
    series_std = inharm.Harmonics(f0=f0_std, jitter_rate=jr, fmin=fmin, fmax=fmax)
    jit_factors = series_std.get_factors()
    series_pos = inharm.Harmonics(f0=f0_pos, jitter_factors=jit_factors, fmin=fmin, fmax=fmax)     

    in_std = inharm.Sound(series_std, length=stim_length)
    in_std.filter(**filter_spec)
    in_std.save(f'{fpath}{f0_std}_{i}.wav')
    
    in_pos = inharm.Sound(series_pos, length=stim_length)
    in_pos.filter(**filter_spec)
    in_pos.save(f'{fpath}{f0_pos}_{i}.wav')
    
# %%

make_tarfile('soundpool_LG.tar.gz', fpath)