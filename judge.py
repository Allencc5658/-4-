import json
import os
import time
from openai import OpenAI

# API配置
QWEN_CHAT_MODEL_API_KEY = "sk-a2fa9edb06b04daa9a0e7af49c546944"
QWEN_CHAT_MODEL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL_NAME = "qwen-max"

PROMPT = """你是一个老师，擅长批改学生的作业。
### 要求
给你一个题目、标准答案、学生答案，判断学生的答案是否正确
你只关注最终的答案是否正确，不关注中间过程


##### 格式
如果标准答案和学生答案的最终答案一样，那么只输出Y
如果标准答案和学生答案的最终答案不一样，那么只输出N


# 题目：{question}
###### 标准答案：{answer}
###### 学生答案：{model_answer}
"""

class QwenChatAPINotStream:
    def ask(self, query: str, key: str):
        client = OpenAI(
            api_key=key,
            base_url=QWEN_CHAT_MODEL_BASE_URL,
        )

        try:
            completion = client.chat.completions.create(
                model=QWEN_CHAT_MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': query}
                ],
            )

            response_data = (json.loads(completion.model_dump_json()))["choices"][-1]["message"]["content"]
            return {"status": 0, "answer": response_data}
        except Exception as e:
            return {"status": -1, "answer": str(e)}

qwen_chat_model_not_stream = QwenChatAPINotStream()

def judge(question: str, answer: str, model_answer: str):
    # 如果 model_answer 为 null，直接返回 N
    if model_answer is None:
        return "N"

    qwen_answer = qwen_chat_model_not_stream.ask(
        query=PROMPT.replace("{question}", question).replace("{answer}", answer).replace("{model_answer}", model_answer),
        key=QWEN_CHAT_MODEL_API_KEY
    )

    if qwen_answer["status"] < 0:
        time.sleep(15)
        qwen_answer = qwen_chat_model_not_stream.ask(
            query=PROMPT.replace("{question}", question).replace("{answer}", answer).replace("{model_answer}", model_answer),
            key=QWEN_CHAT_MODEL_API_KEY
        )
        if qwen_answer["status"] < 0:
            return "E:" + qwen_answer["answer"]

    return qwen_answer["answer"]

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    total_questions = 0
    correct_questions = 0
    results = []

    if isinstance(data, list):
        for item in data:
            question = item.get("instruction", "")
            answer = item.get("answer", "")
            model_answer = item.get("output", None)  # 获取模型答案，如果为 null 则直接返回 N
            score = item.get("score", 0)

            result = judge(question, answer, model_answer)
            results.append({
                "question": question,
                "standard_answer": answer,
                "model_answer": model_answer,
                "judgment": result,
                "score": score
            })

            total_questions += 1
            if result == "Y":
                correct_questions += 1
    else:
        question = data.get("instruction", "")
        answer = data.get("answer", "")
        model_answer = data.get("output", None)  # 获取模型答案，如果为 null 则直接返回 N
        score = data.get("score", 0)

        result = judge(question, answer, model_answer)
        results.append({
            "question": question,
            "standard_answer": answer,
            "model_answer": model_answer,
            "judgment": result,
            "score": score
        })

        total_questions = 1
        if result == "Y":
            correct_questions = 1

    correct_rate = correct_questions / total_questions if total_questions > 0 else 0

    # 添加正确率信息
    results.append({
        "total_questions": total_questions,
        "correct_questions": correct_questions,
        "correct_rate": correct_rate
    })

    return results

def save_result(result, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

def main():
    input_folder = "C:/Users/allen/Desktop/zhihuijiaotong/RAG/output/result_origin"  # 输入文件夹路径
    output_folder = "C:/Users/allen/Desktop/zhihuijiaotong/RAG/output/judge_origin"  # 输出文件夹路径

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取已处理的文件列表
    processed_files = set()
    for file_name in os.listdir(output_folder):
        if file_name.endswith("_result.json"):
            processed_files.add(file_name.replace("_result.json", ".json"))

    # 遍历输入文件夹中的 JSON 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json") and file_name not in processed_files:
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name.replace(".json", "_result.json"))

            try:
                result = process_file(input_file)
                save_result(result, output_file)
                print(f"处理完成：{file_name}，结果已保存到 {output_file}")
            except Exception as e:
                print(f"处理文件 {file_name} 时出错：{str(e)}")

if __name__ == "__main__":
    main()