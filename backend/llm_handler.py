from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMHandler:
    """
    Wrapper for Salamandra (or similar chat-instruct models).
    Handles initialization, prompt formatting, and response generation.
    """
    def __init__(self, config):
        self.config = config
        if not self.config.generator.model_name:
            print("[LLMHandler] No model_name provided → using dummy responses.")
            self.model = None
            self.tokenizer = None
            self.dummy = True
        else:
            self.model, self.tokenizer = self._load_model_and_tokenizer()
            self.dummy = False

        # Default Galician prompts (used in RAG context mode)
        self.default_system_prompt_pre = self._default_prompt_pre()
        self.default_system_prompt_post = self._default_prompt_post()

    # ----------------------------
    # Initialization
    # ----------------------------
    def _load_model_and_tokenizer(self):
        """Load Salamandra model + tokenizer."""
        model_name = self.config.generator.model_name
        cache_dir = self.config.general_config.hf_cache_dir

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            quantization_config=quantization_config if self.generator.quantization else None
        )
        return model, tokenizer

    # ----------------------------
    # Default system prompts
    # ----------------------------
    def _default_prompt_pre(self):
        return (
            "You are a helpful and impartial news assistant that always replies in Galician. \n"
            "You will answer questions using only the information provided in the context below. \n"
            "If the context contains enough information to answer, provide a short, clear summary (3–5 sentences). \n"
            "Respond in a clear and natural tone, similar to a short news brief, but keep your answers concise. \n"
            "Do not add any information that is not present in the context. \n"
            "If the context does not contain the answer, reply exactly with: Non o sei. Esa información non está dispoñible. \n"
            "Do not explain further, do not elaborate, and do not repeat. End your reply immediately after writing that sentence.\n\n"
            "Context:\n"
            "-------------------------\n"
        )

    def _default_prompt_post(self):
        return (
            "-------------------------\n\n"
            "The user will now ask a question related to the context. \n"
            "Answer based only on the provided context."
        )

    # ----------------------------
    # Prompt preparation
    # ----------------------------
    def _prepare_prompt(self, messages):
        """Apply Salamandra chat template to role-based messages."""
        date_string = datetime.today().strftime('%Y-%m-%d')
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            date_string=date_string
        )

    def _tokenize_prompt(self, prompt):
        """Convert text prompt into token IDs on the correct device."""
        return self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

    # ----------------------------
    # Response generation
    # ----------------------------
    def generate_reponse(self, messages, max_new_tokens=200):
        """
        Generate a response for a chat conversation.

        Args:
            messages (list): [{"role": "user"/"assistant"/"system", "content": "..."}]
        """
        if self.dummy:
            return "[DUMMY RESPONSE] No model loaded. Messages received: " + str(messages)

        # Prepare inputs
        prompt = self._prepare_prompt(messages)
        inputs = self._tokenize_prompt(prompt)

        # Generate
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3
        )

        # Decode and clean
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Stop sequence cleanup
        stop_sequence = "Non o sei."
        if stop_sequence in response:
            response = response.split(stop_sequence)[0] + stop_sequence

        return response.strip()
