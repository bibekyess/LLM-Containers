import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("beomi/llama-2-ko-7b")
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained("beomi/llama-2-ko-7b", device_map="auto", torch_dtype=dtype, load_in_8bit=True)
    # model.half().cuda()
    model.eval()
    return tokenizer,model

tokenizer,model = load_tokenizer_and_model()


def generate_text(prompt, max_new_tokens=50, temperature=0.9, top_p=0.97, top_k=10, repetition_penalty=1.15):
    # Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Generate text using the model
    output = model.generate(
        input_ids = input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=6,
        bad_words_ids=[[202],[63],],
        num_return_sequences=1,  # You can adjust the number of generated sequences as needed
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage:
prompt = 'system: You are a question answering system "AskMe" that answers questions about "anything" and supports English and Korean languages.#Do not make up an answer, only use the content if any is found. Keep the answers short.#Only use the korean language.##content'
generated_text = generate_text(prompt, max_new_tokens=100, temperature=0.9, top_p=0.97, top_k=10, repetition_penalty=1.15)
print(generated_text)

def askme_response(user_query, content):
    
    if not isinstance(content, list) or len(content) != 3:
        raise ValueError("Content should be a list containing 3 strings")
    
    combined_content = str(content)

    prompt_format = ('system: You are a question answering system "AskMe" that answers questions about "anything" and '
                     'supports English and Korean languages.#Do not make up an answer, only use the content if any is found. '
                     'Keep the answers short.#Only use the korean language.##content: {content}##user: {query}##AskMe:')
    
    prompt = prompt_format.format(content=combined_content, query=user_query)

    generated_text = generate_text(prompt, max_new_tokens=300, temperature=0.9, top_p=0.97, top_k=20, repetition_penalty=1.07)
    
    response_start_index = generated_text.find("##AskMe:") + len("##AskMe:")
    response = generated_text[response_start_index:].strip().strip("#")
    
    return response

# Example usage:
user_query = "임진왜란이 발생한 년도가 몇년도야?"
content = [
        "임진왜란은 한국 역사에서 중요하게 다뤄지는 전쟁 중 하나입니다.\n 그것은 조선와 일본 사이의 전쟁이었고, 많은 사람들이 피해를 입었습니다.\n 이 전쟁은 1592년에 시작되었으며, 중국 멍골족이 중재하는 1598년까지 계속되었습니다.\n 조선은 이 전쟁에서 심각한 손해를 입었지만 몇 년 동안 전쟁을 끝내는 데 성공했습니다.\n 이 후결과로 조선은 많은 도전에 직면하게 되었지만 국가를 공성하기 위해 견딜 수 있었습니다.\n 이전에는 일별 군사의 임무를 협상하고 있었지만, 이제는 국가의 생존이 위협받았습니다.\n 이러한 도전은 시대를 변화시키고, 국가의 역사를 비관적으로 만들었던 요인입니다.",
        "고구려는 고대 한반도에서 가장 강력한 국가 중 하나였습니다.\n 그들의 군사적 역량은 강력했고, 그들의 효과적인 전략 덕분에 많은 영토를 확장할 수 있었습니다.\n 그들의 장기적인 생존력은 그들의 문화, 경제, 군사력을 통해 고구려가 한반도에서 가장 강해지게 된 원인 중 하나였다.\n 그러나 이러한 강력함에도 불구하고, 결국 다른 나라에 의해 멸망하게 되었습니다.",
        "백제는 고대 한반도에서 가장 중요한 국가 중 하나였습니다.\n 그들은 문화적, 경제적, 군사적 면에서 독특했으며 특히 그들의 예술 성과는 오늘날까지도 매우 인상적입니다.\n 그들의 예술 작품은 섬세하게 조각하고 디테일을 공들인 것으로 유명했다.\n 그들은 또한 뛰어난 군사력과 함께 새로운 영토를 개척하고, 그 장소에서 그들의 문화와 전통을 유지하는 데 성공했다."
    ]

response = askme_response(user_query, content)
print(response)

print("***************************************")

# Example usage:
user_query = "임진왜란이 발생한 년도가 몇년도야?"
content = [
        "임진왜란은 한국 역사에서 중요하게 다뤄지는 전쟁 중 하나입니다.\n 그것은 조선와 일본 사이의 전쟁이었고, 많은 사람들이 피해를 입었습니다.\n 이 전쟁은 1592년에 시작되었으며, 중국 멍골족이 중재하는 1598년까지 계속되었습니다.\n 조선은 이 전쟁에서 심각한 손해를 입었지만 몇 년 동안 전쟁을 끝내는 데 성공했습니다.\n 이 후결과로 조선은 많은 도전에 직면하게 되었지만 국가를 공성하기 위해 견딜 수 있었습니다.\n 이전에는 일별 군사의 임무를 협상하고 있었지만, 이제는 국가의 생존이 위협받았습니다.\n 이러한 도전은 시대를 변화시키고, 국가의 역사를 비관적으로 만들었던 요인입니다.",
        "고구려는 고대 한반도에서 가장 강력한 국가 중 하나였습니다.\n 그들의 군사적 역량은 강력했고, 그들의 효과적인 전략 덕분에 많은 영토를 확장할 수 있었습니다.\n 그들의 장기적인 생존력은 그들의 문화, 경제, 군사력을 통해 고구려가 한반도에서 가장 강해지게 된 원인 중 하나였다.\n 그러나 이러한 강력함에도 불구하고, 결국 다른 나라에 의해 멸망하게 되었습니다.",
        "백제는 고대 한반도에서 가장 중요한 국가 중 하나였습니다.\n 그들은 문화적, 경제적, 군사적 면에서 독특했으며 특히 그들의 예술 성과는 오늘날까지도 매우 인상적입니다.\n 그들의 예술 작품은 섬세하게 조각하고 디테일을 공들인 것으로 유명했다.\n 그들은 또한 뛰어난 군사력과 함께 새로운 영토를 개척하고, 그 장소에서 그들의 문화와 전통을 유지하는 데 성공했다."
    ]

response = askme_response(user_query, content)
print(response)

print("***************************************")

# Example usage:
user_query = "임진왜란이 발생한 년도가 몇년도야?"
content = [
        "임진왜란은 한국 역사에서 중요하게 다뤄지는 전쟁 중 하나입니다.\n 그것은 조선와 일본 사이의 전쟁이었고, 많은 사람들이 피해를 입었습니다.\n 이 전쟁은 1592년에 시작되었으며, 중국 멍골족이 중재하는 1598년까지 계속되었습니다.\n 조선은 이 전쟁에서 심각한 손해를 입었지만 몇 년 동안 전쟁을 끝내는 데 성공했습니다.\n 이 후결과로 조선은 많은 도전에 직면하게 되었지만 국가를 공성하기 위해 견딜 수 있었습니다.\n 이전에는 일별 군사의 임무를 협상하고 있었지만, 이제는 국가의 생존이 위협받았습니다.\n 이러한 도전은 시대를 변화시키고, 국가의 역사를 비관적으로 만들었던 요인입니다.",
        "고구려는 고대 한반도에서 가장 강력한 국가 중 하나였습니다.\n 그들의 군사적 역량은 강력했고, 그들의 효과적인 전략 덕분에 많은 영토를 확장할 수 있었습니다.\n 그들의 장기적인 생존력은 그들의 문화, 경제, 군사력을 통해 고구려가 한반도에서 가장 강해지게 된 원인 중 하나였다.\n 그러나 이러한 강력함에도 불구하고, 결국 다른 나라에 의해 멸망하게 되었습니다.",
        "백제는 고대 한반도에서 가장 중요한 국가 중 하나였습니다.\n 그들은 문화적, 경제적, 군사적 면에서 독특했으며 특히 그들의 예술 성과는 오늘날까지도 매우 인상적입니다.\n 그들의 예술 작품은 섬세하게 조각하고 디테일을 공들인 것으로 유명했다.\n 그들은 또한 뛰어난 군사력과 함께 새로운 영토를 개척하고, 그 장소에서 그들의 문화와 전통을 유지하는 데 성공했다."
    ]

response = askme_response(user_query, content)
print(response)
