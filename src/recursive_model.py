import os
import re
import random
import time
from collections import defaultdict
from PIL import Image
from vlm_model.openai_model import OpenAIModel, encode_image_base64

class RecursiveICLModel(OpenAIModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.chunk_size = kwargs.get("chunk_size", 5)  # Number of images to summarize at once
        
    def prepare_inputs(self, test_item, data):
        """
        Overridden to implement Recursive ICL.
        1. Parse context to separate exemplars.
        2. Group exemplars by label.
        3. Recursively summarize each label's visual features.
        4. Construct a new prompt with summaries + query image.
        """
        full_context = test_item["context"]
        image_path_list = test_item["image_list"]
        
        # The last image is the query image
        query_image_path = image_path_list[-1]
        exemplar_image_paths = image_path_list[:-1]
        
        # Parse labels from context
        # Context format: "<image>\nlabel: {label}\n\n<image>\nlabel: {label}..."
        # We split by <image> to align with exemplar_image_paths
        
        # Simple regex to find all labels
        labels = re.findall(r"label: (.*?)(?:\n|$)", full_context)
        
        if len(labels) != len(exemplar_image_paths):
            print(f"Warning: Number of labels ({len(labels)}) does not match number of exemplar images ({len(exemplar_image_paths)}). Fallback to standard.")
            return super().prepare_inputs(test_item, data)

        # Group by label
        label_to_images = defaultdict(list)
        for img_path, label in zip(exemplar_image_paths, labels):
            label_to_images[label.strip()].append(img_path)
            
        # Recursive Step: Generate descriptions for each label
        label_descriptions = {}
        
        for label, paths in label_to_images.items():
            # Limit paths if too many to save cost/time
            paths_to_use = paths[:self.chunk_size] 
            
            # Construct summary prompt
            # We treat this as a mini-chat
            content = []
            content.append({"type": "text", "text": f"Here are some images of '{label}'. Describe the common visual features of this category in 1-2 sentences."})
            
            for p in paths_to_use:
                try:
                    img = Image.open(p).convert('RGB')
                    b64 = encode_image_base64(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"} # Use low detail for summary to save tokens
                    })
                except Exception as e:
                    print(f"Error loading image {p}: {e}")
                    continue
            
            messages = [{"role": "user", "content": content}]
            
            # Call API to get description
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.5
                )
                description = response.choices[0].message.content
                label_descriptions[label] = description
                # print(f"Generated description for {label}: {description}")
            except Exception as e:
                print(f"Failed to summarize {label}: {e}")
                label_descriptions[label] = "No description available."

        # Final Prompt Construction
        # "You need to recognize entities... Here are descriptions of the candidate classes: ... Now classify this image."
        
        new_context = "Class Descriptions:\n"
        for label, desc in label_descriptions.items():
            new_context += f"- {label}: {desc}\n"
            
        user_template = data["user_template"] # "You need to recognize... \n\nTraining examples:\n{context}\n\nNow classify this image: {question}"
        
        # We replace the original context (lots of images) with our summary context
        # Note: The original template expects {context} to be the training examples. We swap it.
        
        final_text = f"You need to recognize entities in images. Use the provided class descriptions to assign a label.\n\n{new_context}\n\nNow classify this image: <image>"
        
        # Prepare the final message for the query image
        try:
            query_img = Image.open(query_image_path).convert('RGB')
            query_b64 = encode_image_base64(query_img)
            
            final_content = [
                {"type": "text", "text": final_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{query_b64}", "detail": self.image_detail}
                }
            ]
            
            messages = [{"role": "user", "content": final_content}]
            return {"messages": messages}
            
        except Exception as e:
            print(f"Error preparing final input: {e}")
            return {"messages": []} # Should handle error better
