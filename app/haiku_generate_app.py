import streamlit as st
import base64


import random
import sys
import os
sys.path.insert(0, os.getcwd())

#import GRUEN.Main as gruen

st.title("Haiku Stuff!")

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack("images/p1.jpg")


from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("fabianmmueller/deep-haiku-gpt-2")
model = AutoModelForCausalLM.from_pretrained("fabianmmueller/deep-haiku-gpt-2")

topic = st.text_input('Enter a topic!', 'Life')

if st.button("GO!"):

    prompt = "(" + topic.strip()
    if not "=" in prompt:
      prompt += " ="

    with torch.no_grad():
      prompt_tokens = tokenizer(prompt, return_tensors='pt').input_ids
      prompt_outputs = model.generate(prompt_tokens, max_length=40, do_sample=True, num_return_sequences=30, temperature=0.69)

    generated_haikus = []
    processed_generated_haikus = []

    for i, prompt_output in enumerate(prompt_outputs):
      decoded_generated_haiku_doc_ids = (tokenizer.decode(prompt_outputs[i], skip_special_tokens=True))
      haiku_components = decoded_generated_haiku_doc_ids.split(")")
      indiv_haiku = haiku_components[0][1:].strip()
      process_indiv_haiku = indiv_haiku.split(" = ")[1].strip()
      generated_haikus.append(process_indiv_haiku)
      processed_generated_haikus.append(process_indiv_haiku.replace(" / ", " "))

    #quality = gruen.get_gruen(processed_generated_haikus)
    sampled_haiku = random.choice(processed_generated_haikus)

    st.text_area(label ="", value=sampled_haiku, height=50)
