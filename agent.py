import os
from ollama import chat

MODEL='qwen2.5-coder:32b'

def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def create_file(path, content=''):
    """Create a file with the given content."""
    with open(path, 'w') as f:
        f.write(content)

def read_file(path):
    """Read the contents of a file."""
    with open(path, 'r') as f:
        return f.read().strip()

def edit_file(path, new_content):
    """Edit the contents of a file."""
    with open(path, 'w') as f:
        f.write(new_content)

def generate(msg):
    response = chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': msg,
        },
    ])
    return response['message']['content']


from langgraph.graph import Graph

def run_llm_inference():
    # Define the workflow using LangGraph
    graph = Graph()

    # Step 1: Read prompts from files
    gen_prompt = read_file('prompts/tlcode/gen.txt')
    eval_prompt = read_file('prompts/tlcode/eval.txt')
    cute_prompt = read_file('prompts/tlcode/cute.txt')

    # Step 2: Generate TL Sketch
    tl_sketch = graph.add_node("generate_tl_sketch", lambda input: generate(gen_prompt))
    graph.add_edge("generate_tl_sketch", "print_tl_sketch")

    graph.add_node("print_tl_sketch", lambda tl_sketch: print("\033[92mGenerated TL Sketch:\033[0m\n" + tl_sketch))
    graph.add_edge("print_tl_sketch", "eval_prompt_with_sketch")

    # Step 3: Run generation for the evaluation prompt using the generated TL sketch
    eval_output = graph.add_node("eval_prompt_with_sketch", lambda tl_sketch: f"{eval_prompt}\nEval the generated TL Sketch and further improve it:\n{tl_sketch}")
    graph.add_edge("eval_prompt_with_sketch", "generate_eval_output")

    graph.add_node("generate_eval_output", lambda eval_output: generate(eval_output))
    graph.add_edge("generate_eval_output", "print_eval_output")

    graph.add_node("print_eval_output", lambda eval_output: print("\033[94mEvaluation Output:\033[0m\n" + eval_output))
    graph.add_edge("print_eval_output", "final_prompt")

    # Step 4: Process the output with the cute prompt
    graph.add_node("final_prompt", lambda eval_output: f"{cute_prompt}\nGenerate the NVIDIA CuTe code for the given TL Sketch:\n{eval_output}")
    graph.add_edge("final_prompt", "generate_final_output")

    final_output = graph.add_node("generate_final_output", lambda final_prompt: generate(final_prompt))
    graph.add_edge("generate_final_output", "print_final_output")

    graph.add_node("print_final_output", lambda final_output: print("\033[95mFinal Output:\033[0m\n" + final_output))

    graph.set_entry_point("generate_tl_sketch")
    graph.set_finish_point("print_final_output")

    # Execute the workflow
    app = graph.compile()
    app.invoke(input="")
    
if __name__ == "__main__":
    run_llm_inference()
