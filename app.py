import os
import torch
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes
from langdetect import detect


model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
        


def translation(text):
    start_time = time.time()
    #source = flores_codes[source]
    #target = flores_codes[target]
    lang = detect(text)

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=lang , tgt_lang="eng_Latn")
    output = translator(text, max_length=400)
    
    end_time = time.time()

    full_output = output
    output = output[0]['translation_text']
    result = {'inference_time': end_time - start_time,
              'source': source,
              'target': target,
              'result': output,
              'full_output': full_output}
    return result


if __name__ == '__main__':
    print('\tinit models')
    
    # define gradio demo
    lang_codes = list(flores_codes.keys())
    #inputs = [gr.inputs.Radio(['nllb-distilled-600M', 'nllb-1.3B', 'nllb-distilled-1.3B'], label='NLLB Model'),
    inputs = [
              gr.inputs.Textbox(lines=5, label="Input text"),
              ]

    outputs = gr.outputs.JSON()

    title = "NLLB based API"

    demo_status = "Demo is running on CPU"
    description = f"Details: https://github.com/facebookresearch/fairseq/tree/nllb. {demo_status}"
    examples = [
    ['Yue Chinese', 'English', '你食咗飯未?']
    ]

    gr.Interface(translation,
                 inputs,
                 outputs,
                 title=title,
                 description=description,
                 examples=examples,
                 examples_per_page=50,
                 ).launch()
