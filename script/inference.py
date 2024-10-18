
import openai
import time
from dotenv import load_dotenv
import google.generativeai as genai
import os
import anthropic

def run_inference(text, example, args):

    sys_prompt = {"role": "system", "content": "Write a high-quality answer to the given question using only the exact words or phrases from the text.\n"}
    sys_prompt['content'] += "Note that the Text may contain irrelevant information (noise).\n"
    sys_prompt['content'] += "Return only the answer by writing 'Answer: XXX.'\n"

    if example is not None:
        if args.task_name == 'revname' or args.task_name == 'revdesc':
            example = f"Text: {example[0]['text']}\nQuestion: {example[0]['question']}\nAnswer: {example[0]['answer']}.\n" +\
                      f"Text: {example[1]['text']}\nQuestion: {example[1]['question_not']}\nAnswer: {example[1]['answer_not']}."
        else:
            example ="".join([f"Text: {example[k]['text']}\nQuestion: {example[k]['question']}\nAnswer: {example[k]['answer']}.\n" for k in range(args.n_shot)])
        usr_prompt = {"role": "user", "content": example+"\n"}
    else:
        usr_prompt = {"role": "user", "content": ''}


    if args.position == 'qa':
        usr_content = f"Text: {text['no-noise']}\nQuestion: {text['question']}\nAnswer: "
    else:
        usr_content = f"Text: {text['context']}\nQuestion: {text['question']}\nAnswer: "
    usr_prompt['content'] += usr_content

    prompt = [sys_prompt, usr_prompt]
    print(f"length of prompt: {sum([len(p['content'].split()) for p in prompt])}")
    if 'gpt' in args.model:
        openai.api_key = os.environ.get("openai_key")
        try:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=prompt,
                temperature=0.0,
                max_tokens=100,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            time.sleep(60)
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=prompt,
                temperature=0.0,
                max_tokens=100,
            )
            reply = response.choices[0].message.content

    elif 'gemini' in args.model:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        def try_generate(key):
            api_key = os.environ.get(key)
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name=args.model,
                                              system_instruction=sys_prompt['content'])
                responses = model.generate_content(
                    usr_prompt['content'],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=100,
                        temperature=0.0,

                    ),
                    safety_settings=safety_settings
                )
                return responses.text
            except Exception as e:
                print(f"API call failed with {key}: {e}")
                return ''

        reply = try_generate("google_key1")

    elif 'claude' in args.model:
        api_key = os.environ.get("anthropic_key")
        client = anthropic.Anthropic(
            api_key=api_key,
        )
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=100,
                temperature=0.0,
                system=sys_prompt['content'],
                messages=[usr_prompt],
            )
            reply = response.content[0].text
        except Exception as e:
            print(f"API call failed: {e}")
            time.sleep(60)
            response = client.messages.create(
                model=args.model,
                max_tokens=100,
                temperature=0.0,
                system=sys_prompt['content'],
                messages=[usr_prompt],
            )
            reply = response.content[0].text

    sys_prompt_text = '<System Prompt>\n' + sys_prompt['content']
    usr_prompt_text = '<User Prompt>\n' + usr_prompt['content']
    prompt_text = sys_prompt_text + '\n' + usr_prompt_text
    print(prompt_text)
    print(reply)

    return reply, prompt_text


class LLM:
    def __init__(self, model, task_name):
        self.model = model
        self.task_name = task_name
        load_dotenv(verbose=True)

    def get_results(self, text, examples, args):
        prediction,prompt_text = run_inference(text,examples,args)
        return prediction, prompt_text
