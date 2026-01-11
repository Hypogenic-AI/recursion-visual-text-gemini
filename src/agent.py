from src.utils import LLMClient

class BaselineAgent:
    def __init__(self, model="google/gemini-pro-1.5"):
        self.client = LLMClient(model=model)

    def run(self, sample):
        question = sample['question']
        ctxs = sample['ctxs']
        
        # Concatenate all text contexts
        full_context = ""
        for ctx in ctxs:
            if ctx['type'] == 'text':
                full_context += ctx['text'] + "\n\n"
            elif ctx['type'] == 'image':
                full_context += "[IMAGE Placeholder]\n"
        
        prompt = f""".format(
            full_context=full_context,
            question=question
        )
        return self.client.generate(prompt)

class RecursiveAgent:
    def __init__(self, model="google/gemini-pro-1.5", chunk_size=5):
        self.client = LLMClient(model=model)
        self.chunk_size = chunk_size  # Process N context items at a time

    def run(self, sample):
        question = sample['question']
        ctxs = sample['ctxs']
        
        state = "No information gathered yet."
        
        # Recursive/Iterative processing
        # We group ctxs into larger chunks to avoid too many API calls
        batched_ctxs = [ctxs[i:i + self.chunk_size] for i in range(0, len(ctxs), self.chunk_size)]
        
        for i, batch in enumerate(batched_ctxs):
            batch_text = ""
            for ctx in batch:
                if ctx['type'] == 'text':
                    batch_text += ctx['text'] + "\n"
                elif ctx['type'] == 'image':
                    batch_text += "[IMAGE Placeholder]\n"
            
            # Recursion step: Update state
            prompt = f""".format(
                state=state,
                batch_text=batch_text,
                question=question
            )
            state = self.client.generate(prompt, system_prompt="You are a recursive reasoning engine.")
        
        # Final Synthesis
        final_prompt = f""".format(
            state=state,
            question=question
        )
        return self.client.generate(final_prompt)
