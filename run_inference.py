from ollama import chat

MODEL='qwen2.5-coder:32b'
def generate(msg):
    response = chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': msg,
        },
    ])
    return response['message']['content']


def run_llm_inference():
    # Read prompts from files
    with open('prompts/tlcode/gen.txt', 'r') as f:
        gen_prompt = f.read().strip()
    
    with open('prompts/tlcode/eval.txt', 'r') as f:
        eval_prompt = f.read().strip()
    
    with open('prompts/tlcode/cute.txt', 'r') as f:
        cute_prompt = f.read().strip()

    # Step 1: Generate TL Sketch
    tl_sketch = generate(gen_prompt)
    print("\033[92mGenerated TL Sketch:\033[0m")
    print(tl_sketch)

    # Step 2: Run generation for the evaluation prompt using the generated TL sketch
    eval_prompt_with_sketch = f"{eval_prompt}\nEval the generated TL Sketch and further improve it:\n{tl_sketch}"
    eval_output = generate(eval_prompt_with_sketch)
    print("\033[94mEvaluation Output:\033[0m")
    print(eval_output)

    # Step 3: Process the output with the cute prompt
    final_prompt = f"{cute_prompt}\nGenerate the NVIDIA CuTe code for the given TL Sketch:\n{eval_output}"
    final_output = generate(final_prompt)
    print("\033[95mFinal Output:\033[0m")
    print(final_output)

    print("Final Output:", final_output)

if __name__ == "__main__":
    run_llm_inference()
