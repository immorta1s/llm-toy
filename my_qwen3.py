import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM 

class MyQwen3(LLM):
    
    def __init__(self, model_path: str = "./Qwen3-8B/", device: str = "cpu", do_think=False, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._device = device
        self._do_think = do_think
        self._thinking_content = ''
        
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
            enable_thinking=self._do_think, # Switches between thinking and non-thinking modes. Default is True.
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

        self._thinking_content = self._tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        #self._thinking_content = thinking_content
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return content

    @property
    def _llm_type(self) -> str:
        return "qwen3-8b(local)"
    
    @property
    def show_thinking(self) -> str:
        cleaned_text = self._thinking_content.replace("<think>", "").replace("</think>", "").strip()
        return cleaned_text
    

if __name__ == '__main__':
    # 实例化你的本地模型
    #local_llm = MyQwen3(model_path="./Qwen3-8B/")
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    local_llm = MyQwen3(do_think=True)
    # 测试 invoke 调用
    input_text = "我现在有500，每年增加5%,10年后我有多少钱？"
    result = local_llm.invoke(input_text)
    #promptTemplate = ChatPromptTemplate([
    #('system', '你是一个周游世界的旅行家。请根据用户的问题回答。'),
    #('user', '这是用户的问题 {topic}')
    #])
    #chain = promptTemplate | local_llm
    #input_text = '帮我介绍一下香港的风土人情'
    #result = chain.invoke(input_text)
    print('========= 思考过程 =========')
    print(local_llm.show_thinking)
    print('============= 模型输出最终答案 ==============')
    print(result)