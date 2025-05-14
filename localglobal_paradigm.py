# The local global paradigm using functions from lg_works_funcs.py and lg_works.py
# Condition 1: Harmonic condition
# Condition 2: Jittered condition
# Condition 3: Jittered condition with shuffled sequences

# %%
import numpy as np
#import serial
from psychopy import core, visual, sound, event, prefs, gui
import pandas as pd
# %%
prefs.hardware["audioLib"] = ["PTB"]
prefs.hardware["audioLatencyMode"] = 4

# %%
# Logger configuration

logger_config = {"fpath": "logs/", "write_to_csv": True}
logdata = []

def init_logger(pid: int) -> dict:
    """
    Initialize the logger using participant ID.
    
    Args:
        pid (int): Participant ID.
    
    Returns:
        dict: key ("pid"): value(pid).
    """
    
    return {"pid": pid}

def add_log(log_dict: dict, pid: int) -> None:
    """
    Add a log entry.
    
    Args:
        log_dict (dict): Log entry to add.
        pid (int): Participant ID.
    """
    log_dict["pid"] = pid
    log_dict["abs_time"] = core.getAbsTime()
    log_dict["time"] = core.getTime()
    logdata.append(log_dict)

def save_log(pid: int) -> None:
    """
    Save the log to a CSV file.
    
    Args:
        pid (int): Participant ID.
    """
    df = pd.DataFrame(logdata)
    df.to_csv(f"{logger_config['fpath']}{pid}_log.csv", index=False)

def save_demography (pid: int, age: int, gender: str) -> None:
    """
    Save demographic information to a CSV file.
    
    Args:
        pid (int): Participant ID.
        age (int): Age of the participant.
        gender (str): Gender of the participant.
    """
    df = pd.DataFrame(
        {
            "pid": [pid],
            "age": [age],
            "gender": [gender],
            "abs_time": [core.getAbsTime()]
        }, index = [0])
    
    df.to_csv(f"{logger_config['fpath']}{pid}_demography.csv")

# %%
config = {
    "soundpool_path": "soundpool/",  # Path to sound files
    "ioi": 0.15,  # Inter-onset interval in seconds
    "inter_sequence_interval": (0.7, 1),  # Random interval between sequences
    "seqs_per_block": 30,  # Number of sequences in the block
}
# %%

import numpy as np

freqs = [300, 350]

rng = np.random.default_rng()
jit_pool = rng.choice(np.arange(1, 10), size=5, replace=False)

# %%

block_probabilities = {
    "xX": {"regularseq": 0.75, "irregularseq": 0.125, "omissionseq": 0.125},
    "xY": {"regularseq": 0.125, "irregularseq": 0.75, "omissionseq": 0.125},
    "xO": {"regularseq": 0.125, "irregularseq": 0.125, "omissionseq": 0.75}
    }

def get_block_probabilities(block_type: str) -> dict:
    """
    Get the probabilities for a given block type, including prime blocks.

    Args:
        block_type (str): The type of block ('xX', 'xX_prime', etc.).

    Returns:
        dict: A dictionary of probabilities for the block.
    """

    base_block = block_type.replace("_prime", "")
    return block_probabilities[base_block]


# %%

def generate_std_dev_harmonic(block_type: str, freqs: list) -> tuple:
    """
    Mapping standard and deviant sounds for harmonic condition (cond1)

    Args:
        block_type (str): The type of block ('xX', 'xX_prime', etc.).
        freqs (list): List of frequencies.

    Returns:
        tuple:
            - str: standard
            - str: deviant
    """
    
    standard = f"{freqs[0]}_0"
    deviant = f"{freqs[1]}_0"
    
    if "prime" in block_type:
        standard, deviant = deviant, standard
    
    return standard, deviant


# %%

def generate_std_dev(block_type: str, jit_pool: list, freqs: list, is_shuffled: bool = False) -> tuple:
    """
    Mapping standard and deviant sounds for jittered conditions (cond2 and cond3)

    Args:
        block_type (str): The type of block ('xX', 'xX_prime', etc.).
        jit_pool (list): The jitter pool to use.
        freqs (list): List of frequencies.

    Returns:
        tuple:
            - list of standards
            - str: deviant
    """
    
    if is_shuffled:
        
        shuffled_jit_pool = rng.permutation(jit_pool)
        
    else:
        
        shuffled_jit_pool = jit_pool
        
    standard = [f"{freqs[0]}_{jit}" for jit in shuffled_jit_pool]
    deviant = f"{freqs[1]}_{shuffled_jit_pool[-1]}"
    
    if "prime" in block_type:
        standard = [f"{freqs[1]}_{jit}" for jit in shuffled_jit_pool]
        deviant = f"{freqs[0]}_{shuffled_jit_pool[-1]}"
        
    return standard, deviant

# %%

def select_seq_type(seq_type: str, standard: list, deviant: str) -> list:
    """ Select sequence for probabilistic sequences 

    Args:
        seq_type (str): The type of sequence ('regularseq', 'irregularseq', 'omissionseq').
        standard (list): List of standards from 'generate_std_dev()'.
        deviant (str): Deviant from 'generate_std_dev()'.

    Returns:
        list: Selected sequence.
    """
    
    if seq_type == "regularseq":
        return standard
    elif seq_type == "irregularseq":
        return standard[:4] + [deviant]
    elif seq_type == "omissionseq":
        return standard[:4] + ["omi"]
    

# %%    

def generate_seqs(block_type: str,
                  n_seq: int,
                  jit_pool: list,
                  freqs: list,
                  is_probabilistic: bool = False,
                  is_cond1: bool = False,
                  is_shuffled: bool = False) -> list:
    
    """
    Generate sequences (fixed or probabilistic) for a block type.
    If probabilistic, you need to provide the block_probabilities (default is None).
    
    Args:
        block_type (str): The type of block to generate ('xX', 'xX_prime', etc.).
        n_seq (int): Number of sequences to generate.
        jit_pool (list): The jitter pool to use.
        freqs (list): List of frequencies.
        is_cond1 (bool, optional): Whether the block is for condition 1. Defaults to True
        is_probabilities (bool, optional): If True, generate probabilistic sequences.
        is_shuffled (bool, optional): Crucial for cond3 to shuffle jitter patterns in sequences. Defaults to False.
    
    Returns:
        list: A list of sequences.
    """
    
    sequences = []
            
    for _ in range(n_seq):
        
        if is_cond1:
            standard_str, deviant = generate_std_dev_harmonic(block_type, freqs)
            
            # in order to use the same logic as in the 2nd and 3rd condition
            # we need to create a list of 5 elements from a string
            standard = [standard_str] * 5
            
        else:
            standard, deviant = generate_std_dev(block_type, jit_pool, freqs, is_shuffled)
        
        if is_probabilistic:
            
            probabilities = get_block_probabilities(block_type)
            
            seq_type = rng.choice(
                ["regularseq", "irregularseq", "omissionseq"],
                p=list(probabilities.values())
            )
            
            sequence = select_seq_type(seq_type, standard, deviant)
            
        else:
            patterns = {
                "xY": standard[:4] + [deviant],
                "xO": standard[:4] + ["omi"],
                "xX": standard,
            }
            
            sequence = patterns.get(block_type.replace("_prime", ""))
            
        sequences.append(sequence)
        
    return sequences

# %%

# functino to generate block using generate_seqs()

def parse_seqs_into_block(block_type: str,
                   n_fix_seq: int = 10,
                   n_prob_seq: int = 20,
                   jit_pool: list = None,
                   freqs: list = None,
                   is_cond1: bool = False,
                   is_shuffled: bool = False) -> list:
    
    """ Generate a full block (fixed + probabilistic) using 'generate_seq()' for a given block type.

    Args:
        block_type (str): Type of block to generate ('xX', 'xX_prime', etc.).
        n_fix_seq (int: Number of fixed sequences to create. Defaults to 10.
        n_prob_seq (int): Number of prob sequences to create. Defaults to 40.
        jit_pool (list): Jitter numbers corresponding to generated soundpool. Defaults to None.
        freqs (list): F0's for sdtandard and deviant. Defaults to None.
        is_cond1 (bool): Is the block being created for harmonic condition?
                Important parameter for the output, as logic for cond1 vs cond2 and cond3 differs. Defaults to False.
        is_shuffled (bool): Crucial for cond3 to shuffle jitter patterns in sequences. Defaults to False.

    Returns:
        list: Concatenated list of fixed and probabilistic sequences.
    """
    
   
    fixed_sequences = generate_seqs(
        block_type=block_type,
        n_seq=n_fix_seq,
        jit_pool=jit_pool,
        freqs=freqs,
        is_cond1=is_cond1,
        is_shuffled=is_shuffled
    )
    
    probabilistic_sequences = generate_seqs(
        block_type=block_type,
        n_seq=n_prob_seq,
        jit_pool=jit_pool,
        freqs=freqs,
        is_cond1=is_cond1,
        is_probabilistic=True,
        is_shuffled=is_shuffled
    )
    
    return fixed_sequences + probabilistic_sequences
# %%

def make_block(block_type: str, condition: int) -> list:
    
    if condition == 1:
        return parse_seqs_into_block(block_type, freqs=freqs, is_cond1=True)
    elif condition == 2:
        return parse_seqs_into_block(block_type, jit_pool=jit_pool, freqs=freqs)
    elif condition == 3:
        return parse_seqs_into_block(block_type, jit_pool=jit_pool, freqs=freqs, is_shuffled=True)

# %%
# Running the experiment

def play_sound(win, sound_object) -> None:
    """
    This function plays a sound from a sound pool on the next flip.
    It takes a sound object as input and uses the `win.getFutureFlipTime()` function
    to determine the next flip time. The sound is then played using the `sound.play()` method 
     
    Args:
        sound (object): The sound object to play   
    Returns:
        None
    """
    next_flip = win.getFutureFlipTime(clock="ptb")
    sound_object.play(when=next_flip)


# Keep track of played blocks
played_blocks = []

def run_all_blocks() -> None:
    """
    Psychopy run
    """
    rng = np.random.default_rng()
     
    conditions = [1, 2, 3]
    block_types = ["xX", "xX_prime", "xY", "xY_prime", "xO", "xO_prime"]
    all_blocks = [(cond, block) for cond in conditions for block in block_types]
    
    rng.shuffle(all_blocks)

    # ID box
    dlg = {"pid": np.random.randint(1000, 10000), "gender": "", "age": ""}
    gui.DlgFromDict(dlg, title="Demography", show=True)
    pid = dlg["pid"]
    
    #Init logger
    #logger = init_logger(pid)
    
    #Save demography
    save_demography(pid, dlg["age"], dlg["gender"])

    # Create PsychoPy window
    win = visual.Window([800, 600], color="black", fullscr=False)
    message = visual.TextBox2(
        win, text="", pos=(0, 0), letterHeight=0.05, alignment="center", autoDraw=True
    )

    for condition, block_type in all_blocks:
        if (condition, block_type) in played_blocks:
            continue
        
        add_log({"event": "block_start", "block_type": block_type, "condition": condition}, pid)
        
        # Generate sequences for the block
        print("###############################")
        print(played_blocks)
        print(len(played_blocks))
        
        sequences = make_block(block_type, condition)

        # Show start message
        message.text = f"Starting block {block_type} from condition {condition}. Press space to begin."
        win.flip()
        event.waitKeys(keyList=["space"])

        # Play each sequence
        for seq_idx, sequence in enumerate(sequences):
            for sound_idx, sound_file in enumerate(sequence):
                # Load and play sound
                sound_path = config["soundpool_path"] + sound_file + ".wav"
                sobj = sound.Sound(sound_path, hamming=False, stereo=True)
                play_sound(win, sobj)
                
                # Update message
                message.text = f"""
                Playing block {block_type} from condition {condition}.\n
                Sequence {seq_idx + 1} of {len(sequences)}.\n
                Sound {sound_idx + 1} of {len(sequence)}.\n
                File: {sound_file}.
                """
                print(f"This is block {block_type} from condition {condition}.")
                print(f"This is sequence {seq_idx + 1} of {len(sequences)}.")
                print(f"This is sound {sound_idx + 1} of {len(sequence)}.")
                print(f"File: {sound_file}.")
                
                #Log
                add_log(
                    {
                        "event": "sound_played",
                        "block_type": block_type,
                        "condition": condition,
                        "sequence_index": seq_idx,
                        "sound_index": sound_idx,
                        "sound_file": sound_file,
                    },
                    pid)
                
                win.flip()

                # Wait for inter-onset interval
                core.wait(config["ioi"])

            # Wait for inter-sequence interval
            # symbol * is used only because the value in dict is a tuple that need to be unpacked and
            # assigned as different params
            inter_sequence_interval = rng.uniform(*config["inter_sequence_interval"])
            core.wait(inter_sequence_interval)

        # Save Log
        save_log(pid)
        
        # Mark block as played
        played_blocks.append((condition,block_type))

        # Show end message for the block
        message.text = f"Block {block_type} finished. Press space to continue."
        win.flip()
        event.waitKeys(keyList=["space"])

    # Show final message
    message.text = "All blocks finished. Press space to exit."
    win.flip()
    event.waitKeys(keyList=["space"])

    # Close PsychoPy window
    win.close()
    core.quit()

# Run all blocks
run_all_blocks()