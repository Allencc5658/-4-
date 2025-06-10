import asyncio
import aiohttp
import json
import os
from tqdm import tqdm

# API 配置
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"# 替换为实际模型名称
CHUNK_SIZE = 10  # 每块数据的大小
TIMEOUT = 600  # 设置超时时间为 60 秒
RETRIES = 3  # 重试次数

# 输入输出文件夹配置
INPUT_FOLDER = "C:/Users/allen/Desktop/zhihuijiaotong/RAG/output/test"  # 包含 JSON 文件的文件夹
OUTPUT_FOLDER = "C:/Users/allen/Desktop/zhihuijiaotong/RAG/output/result_origin"  # 保存结果的文件夹

async def call_qwen_api(instruction, retry_count=0):
    """调用 Qwen 模型 API，包含重试机制"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "12345678"  # 替换为实际 API 密钥
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": instruction}],
        "temperature": 0.7,  # 可根据需要调整
        "max_length": 200  # 可根据需要调整
    }

    try:
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
            async with session.post(API_URL, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    print(f"Error: {response.status}")
                    return None
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        if retry_count < RETRIES:
            print(f"Request failed, retrying ({retry_count + 1}/{RETRIES})...")
            return await call_qwen_api(instruction, retry_count + 1)
        else:
            print(f"Request failed after {RETRIES} retries: {e}")
            return None


async def process_chunk(chunk):
    """处理一个数据块"""
    results = []
    for item in tqdm(chunk, desc="Processing"):
        # 合并 instruction 和 input 作为模型输入
        instruction = item['instruction']
        input_content = item.get('input', '')
        model_input = f"{instruction} {input_content}"

        response = await call_qwen_api(model_input)

        # 保存原始的 input 作为 instruction
        saved_instruction = input_content

        # 保存原始的 output 作为 answer
        answer = item.get("output", "")

        results.append({
            "instruction": saved_instruction,
            "answer": answer,
            "score": item.get("score", 1),
            "output": response
        })
    return results


def read_input_data(file_path):
    """读取输入 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_output_data(data, file_path):
    """保存输出 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_progress(file_path, progress):
    """保存进度"""
    progress_file = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_progress.json")
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)


def load_progress(file_path):
    """加载进度"""
    progress_file = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


async def process_file(file_path):
    """处理单个文件，支持断点续传"""
    # 读取输入数据
    input_data = read_input_data(file_path)

    # 加载进度
    progress = load_progress(file_path)
    if progress:
        start_index = progress.get('start_index', 0)
        output_data = progress.get('output_data', [])
    else:
        start_index = 0
        output_data = []

    # 分块处理数据
    total_data = len(input_data)

    for i in range(start_index, total_data, CHUNK_SIZE):
        chunk = input_data[i:i + CHUNK_SIZE]
        results = await process_chunk(chunk)
        output_data.extend(results)

        # 保存当前块的结果
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file_name)[0]}_result.json")
        save_output_data(output_data, output_file_path)

        # 保存进度
        progress_data = {
            'start_index': i + CHUNK_SIZE,
            'output_data': output_data
        }
        save_progress(file_path, progress_data)

        print(f"Processed {i + CHUNK_SIZE} of {total_data} items. Saved to {output_file_path}")

    # 处理完成后删除进度文件
    progress_file = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path)}_progress.json")
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"Processing completed for {file_path}. Results saved to {output_file_path}")


async def main():
    # 获取输入文件夹中的所有 JSON 文件
    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]

    for file_name in json_files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        print(f"Processing file: {file_path}")
        await process_file(file_path)


if __name__ == "__main__":
    asyncio.run(main())