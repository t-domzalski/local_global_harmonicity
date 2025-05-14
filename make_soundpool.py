import numpy as np
import inharmonicon.inharmonicon as inharm
from progress.bar import Bar
import tarfile
import os.path

# function definitions
def make_tarfile(output_filename, source_dir):
    print('Compressing to tar...')
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print('Done.')

# stimulus length in seconds
stim_length = .07

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
no_inharmonic_sounds = 1000

# filepath to save output
fpath = 'soundpool/'

# make harmonic sounds
print("Generating harmonic sounds...")

# make harmonic series
hs_std, hs_dev_pos, hs_dev_neg = (inharm.Harmonics(300, fmin=fmin, fmax=fmax),
                                  inharm.Harmonics(350, fmin=fmin, fmax=fmax),
                                  inharm.Harmonics(250, fmin=fmin, fmax=fmax)
                                  )

# generate standard
harm_std = inharm.Sound(f=hs_std, length=stim_length)
harm_std.filter(**filter_spec)
harm_std.save(fpath + 'harm_std.wav')

# pitch oddballs
harm_pitch_pos = inharm.Sound(f=hs_dev_pos, length=stim_length)
harm_pitch_pos.filter(**filter_spec)
harm_pitch_pos.save(fpath + 'harm_pitch_pos.wav')

harm_pitch_neg = inharm.Sound(f=hs_dev_neg, length=stim_length)
harm_pitch_neg.filter(**filter_spec)
harm_pitch_neg.save(fpath + 'harm_pitch_neg.wav')

# intensity oddballs
harm_int_pos = inharm.Sound(f=hs_std, length=stim_length)
harm_int_pos.adjust_volume(+10)
harm_int_pos.filter(**filter_spec)
harm_int_pos.save(fpath + 'harm_int_pos.wav')

harm_int_neg = inharm.Sound(f=hs_std, length=stim_length)
harm_int_neg.adjust_volume(-10)
harm_int_neg.filter(**filter_spec)
harm_int_neg.save(fpath + 'harm_int_neg.wav')

# location oddballs
harm_loc_pos = inharm.Sound(f=hs_std, length=stim_length)
harm_loc_pos.filter(**filter_spec)
harm_loc_pos.apply_itd(800, 'right')
harm_loc_pos.save(fpath + 'harm_loc_pos.wav')

harm_loc_neg = inharm.Sound(f=hs_std, length=stim_length)
harm_loc_neg.filter(**filter_spec)
harm_loc_neg.apply_itd(800, 'left')
harm_loc_neg.save(fpath + 'harm_loc_neg.wav')

# omission oddball
harm_omission = inharm.Sound(None, length=stim_length)
harm_omission.save(fpath + 'omission.wav')

# make inharmonic sounds
print(f"Generating {no_inharmonic_sounds} sounds...")
for i in Bar('Processing').iter(range(no_inharmonic_sounds)):
    # inharmonic series
    series_std = inharm.Harmonics(300, jitter_rate=jr, fmin=fmin, fmax=fmax)
    jit_factors = series_std.get_factors()
    series_dev_pos = inharm.Harmonics(350, jitter_factors=jit_factors, fmin=fmin, fmax=fmax)
    series_dev_neg = inharm.Harmonics(250, jitter_factors=jit_factors, fmin=fmin, fmax=fmax)

    # inharmonic standard
    son = inharm.Sound(series_std, length=stim_length)
    son.filter(**filter_spec)
    son.save(f'{fpath}ih_std_{i}.wav')

    # pitch deviants
    son = inharm.Sound(series_dev_pos, length=stim_length)
    son.filter(**filter_spec)
    son.save(f'{fpath}ih_pitch_pos_{i}.wav')

    son = inharm.Sound(series_dev_neg, length=stim_length)
    son.filter(**filter_spec)
    son.save(f'{fpath}ih_pitch_neg_{i}.wav')

    # intensity deviants
    son = inharm.Sound(series_std, length=stim_length)
    son.adjust_volume(+10)
    son.filter(**filter_spec)
    son.save(f'{fpath}ih_int_pos_{i}.wav')
    son.adjust_volume(-20)
    son.save(f'{fpath}ih_int_neg_{i}.wav')

    # location deviants
    son = inharm.Sound(series_std, length=stim_length)
    son.filter(**filter_spec)
    son.apply_itd(800, 'right')
    son.save(f'{fpath}ih_loc_pos_{i}.wav')

    son = inharm.Sound(series_std, length=stim_length)
    son.filter(**filter_spec)
    son.apply_itd(800, 'left')
    son.save(f'{fpath}ih_loc_neg_{i}.wav')

# make tarball of the soundpool



make_tarfile('soundpool.tar.gz', 'soundpool')
