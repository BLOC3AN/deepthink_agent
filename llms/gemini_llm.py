from langchain_google_genai import ChatGoogleGenerativeAI



class LLMGemini:
    def __init__(
            self, 
            model_name="gemini-2.0-flash", 
            temperature=0.95, 
            top_p=0.7, 
            top_k=50, 
            max_tokens=None, 
            max_output_tokens=None, 
            verbose=True, 
            disable_streaming=True):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            verbose=verbose,
            disable_streaming=disable_streaming,
        )
