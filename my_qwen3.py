import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM 

class MyQwen3(LLM):
    
    def __init__(self, model_path: str = "./Qwen3-8B/", device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._device = device
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
        self._model.eval()
        if torch.cuda.is_available():
            self._model.to("cuda")

    def _call(self, prompt: str, stop: list = None) -> str:
        #inputs = self._tokenizer(prompt, return_tensors="pt").to('cuda')
        messages = [{"role": "user", "content": prompt},]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to('cuda')
        outputs = self._model.generate(
            #inputs["input_ids"],
            **model_inputs,
            max_new_tokens=100000,
            do_sample=False
        )
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist() 
        # parse thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self._tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return content

    @property
    def _llm_type(self) -> str:
        return "qwen3-8b(local)"
    
    

if __name__ == '__main__':
    # 实例化你的本地模型
    #local_llm = MyQwen3(model_path="./Qwen3-8B/")
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    local_llm = MyQwen3()
    # 测试 invoke 调用
    #input_text = "介绍一下你自己。"
    #result = local_llm.invoke(input_text)
    promptTemplate = ChatPromptTemplate([
    ('system', '你是一个周游世界的旅行家。请根据用户的问题回答。'),
    ('user', '这是用户的问题 {topic}')
    ])
    chain = promptTemplate | local_llm
    input_text = '帮我介绍一下香港的风土人情'
    result = chain.invoke(input_text)
    print(result)