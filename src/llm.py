from openai import OpenAI
from dotenv import load_dotenv
import os
import time
_ = load_dotenv()
api_key = os.environ["NVIDIA_API_KEY"]


def get_llm_output(temperature, max_tokens, system_role, prompt,
                   frequency_penalty=0, presence_penalty=0, stop=None):
    """
    Gọi LLM và trả về nội dung response. Nếu system_role là chuỗi rỗng,
    ta chỉ đưa message user, không đưa message system.
    """
    start_time = time.time()
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
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