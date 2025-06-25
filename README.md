# tlcute-attention
Open source implementation of the QiMeng-Attention paper: https://www.arxiv.org/pdf/2506.12355

![alt text](image-6.webp "A cute treasure found by a llama")

# Introduction
The use of TL-Code, an intermediate and high level natural language programming language that is used by LLMs to implement in an easier manner the CuTe implementation of the Attention kernels. This could be used to implement other operations different to attention, and the different attention flavors.

# Run
WIP! At the moment, the simple workflow explained by the paper is implemented in run_inference.py; an agent version of this is in the making in agent.py with the objective of adding looping in the different phases to enhance further (not just 1 step).
So, "python3 run_inferenc.py" would make it. Only dep is ollama, something that we can scrape easily and do requests in the future if necessary.
