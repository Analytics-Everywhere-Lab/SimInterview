from openai import OpenAI
from dotenv import load_dotenv
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
import time
from config import NVIDIA_API_KEY, OPENAI_API_KEY
# Load environment variables
_ = load_dotenv()
api_key_nvi = NVIDIA_API_KEY
api_key = OPENAI_API_KEY

# Hàm gọi LLM để lấy nội dung response
def get_llm_output(temperature, max_tokens, system_role, prompt,
                   frequency_penalty=0, presence_penalty=0, stop=None, llm_type="nvidia"):
    """
    Gọi LLM và trả về nội dung response. Nếu system_role là chuỗi rỗng,
    ta chỉ đưa message user, không đưa message system.
    """
    start_time = time.time()
    if llm_type == "openai":
        # Sử dụng OpenAI API
        client = OpenAI(
            api_key=api_key,
        )
        # Chỉ thêm system message khi system_role không rỗng
        messages = []
        if system_role and system_role.strip():
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        end_time = time.time()
        print('Finish generating response in', round((end_time - start_time), 2), 'seconds')
        return completion.choices[0].message.content
    else:
        # Sử dụng NVIDIA API
        # Chỉ sử dụng NVIDIA API nếu llm_type là "nvidia"
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key_nvi,
        )
        # Chỉ thêm system message khi system_role không rỗng
        messages = []
        if system_role and system_role.strip():
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        end_time = time.time()
        print('Finish generating response in', round((end_time - start_time), 2), 'seconds')
        return completion.choices[0].message.content

# Model để phân tích cú pháp câu hỏi phỏng vấn
class QuestionBank(BaseModel):
    questions: List[str] = Field(descriptions="List of interview questions")

def get_llm_output_with_parser(max_tokens, system_role, prompt,
                                output_model=QuestionBank, llm_type="openai"):
    """
    Gọi LLM và trả về nội dung response đã được phân tích cú pháp.
    """
    parser = PydanticOutputParser(pydantic_object=output_model)
    prompt_template = PromptTemplate(
        input_variables=["system_role", "prompt"],
        template= system_role + "\n" + prompt
    )

    if llm_type == "openai":
        llm = ChatOpenAI(temperature=0, max_tokens=max_tokens, model="gpt-4o-mini", api_key=api_key)
    else:
        llm = ChatNVIDIA(temperature=0, max_tokens=max_tokens, model="meta/llama-3.3-70b-instruct", api_key=api_key_nvi)
    chain = prompt_template | llm | parser
    response = chain.invoke({
        "system_role": system_role,
        "prompt": prompt
    })
    return response.questions