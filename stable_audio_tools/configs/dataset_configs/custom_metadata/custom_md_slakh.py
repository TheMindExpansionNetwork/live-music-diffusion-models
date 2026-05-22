import pandas as pd
import random
def get_custom_metadata(info, audio):
    
    prompt = info['prompt']
    # randomly lowercase some of the prompt to add some variability
    if random.random() < 0.5:
        prompt = prompt.lower()

    return {"prompt": prompt}