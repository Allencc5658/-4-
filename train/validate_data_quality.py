import json
import re
import hashlib
from collections import Counter
import random

def load_data(file_path):
    """加载驾考题目数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"题目数: {len(data)}")
    return data

def normalize_question(text):
    """标准化题目文本，去除多余空格、标点符号差异等"""
    # 去除空格
    text = re.sub(r'\s+', '', text)
    # 统一标点符号
    text = re.sub(r'，', ',', text)
    text = re.sub(r'。', '.', text)
    text = re.sub(r'；', ';', text)
    text = re.sub(r'：', ':', text)
    text = re.sub(r'？', '?', text)
    text = re.sub(r'！', '!', text)
    return text.lower()

def calculate_text_hash(text):
    """计算文本的哈希值，用于快速比较"""
    normalized_text = normalize_question(text)
    return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

def extract_question_text(input_text):
    """提取题目的文本内容，清除题型、选项等信息"""
    # 提取"题目："之后的内容
    match = re.search(r"题目：(.+?)(?:\n|$)", input_text)
    if match:
        question = match.group(1)
        # 去除图片描述和选项
        question = re.sub(r"图片描述：.*", "", question)
        question = re.sub(r"选项[A-D]：.*", "", question)
        return question.strip()
    return input_text.strip()

def extract_answer(output_text):
    """从输出文本中提取答案"""
    # 尝试匹配"正确答案是选项X"模式
    match = re.search(r"正确答案是[：:]\s*([A-D])", output_text)
    if match:
        return match.group(1)
    
    # 尝试匹配"正确"或"错误"
    match = re.search(r"正确答案是[：:]\s*(正确|错误)", output_text)
    if match:
        return "A" if match.group(1) == "正确" else "B"
    
    # 尝试匹配"因此答案选择 X"模式
    match = re.search(r"因此答案选择\s*([A-D])", output_text)
    if match:
        return match.group(1)
    
    return None

def check_duplicates(data):
    """检查是否还有重复题目"""
    question_texts = []
    question_hashes = []
    
    for item in data:
        question_text = extract_question_text(item["input"])
        question_texts.append(question_text)
        question_hashes.append(calculate_text_hash(question_text))
    
    # 检查哈希值重复
    hash_count = Counter(question_hashes)
    duplicates = [(h, count) for h, count in hash_count.items() if count > 1]
    
    print(f"\n检查重复题目:")
    if duplicates:
        print(f"发现 {len(duplicates)} 组重复题目!")
        
        # 输出部分重复示例
        for i, (hash_val, count) in enumerate(duplicates[:5]):
            print(f"\n重复组 {i+1} (重复 {count} 次):")
            
            # 查找具有相同哈希值的题目
            indices = [j for j, h in enumerate(question_hashes) if h == hash_val]
            for idx in indices:
                print(f"- 题目: {question_texts[idx][:50]}...")
    else:
        print("没有发现重复题目，去重成功!")
    
    return len(duplicates)

def analyze_content_completeness(data):
    """分析数据完整性"""
    issues = 0
    
    # 检查所有必要字段是否存在
    for i, item in enumerate(data):
        if "instruction" not in item or not item["instruction"]:
            print(f"警告: 第 {i} 题缺少 instruction 字段")
            issues += 1
        
        if "input" not in item or not item["input"]:
            print(f"警告: 第 {i} 题缺少 input 字段")
            issues += 1
            
        if "output" not in item or not item["output"]:
            print(f"警告: 第 {i} 题缺少 output 字段")
            issues += 1
    
    # 检查input字段是否包含题目内容
    has_question_count = 0
    for item in data:
        if "input" in item and "题目：" in item["input"]:
            has_question_count += 1
    
    question_coverage = has_question_count / len(data) * 100 if data else 0
    print(f"\n题目内容完整性: {question_coverage:.2f}% 的题目包含'题目:'标记")
    
    # 检查output字段是否包含答案和解释
    answer_count = 0
    explanation_count = 0
    
    for item in data:
        if "output" in item:
            output = item["output"]
            
            # 检查是否有答案
            if extract_answer(output) is not None:
                answer_count += 1
            
            # 检查是否有解释（简单检查长度）
            if len(output) > 50:
                explanation_count += 1
    
    answer_coverage = answer_count / len(data) * 100 if data else 0
    explanation_coverage = explanation_count / len(data) * 100 if data else 0
    
    print(f"答案完整性: {answer_coverage:.2f}% 的题目包含可识别的答案")
    print(f"解释完整性: {explanation_coverage:.2f}% 的题目包含足够长度的解释")
    
    return issues

def sample_questions(data, n=5):
    """从数据集中随机抽取样本题目"""
    samples = random.sample(data, min(n, len(data)))
    
    print(f"\n随机抽取 {len(samples)} 题样本:")
    
    for i, item in enumerate(samples):
        question_text = extract_question_text(item["input"])
        answer = extract_answer(item["output"]) if "output" in item else None
        
        print(f"\n样本 {i+1}:")
        print(f"题目: {question_text}")
        print(f"答案: {answer}")
        print(f"解释: {item['output'][:100]}..." if "output" in item else "无解释")

def main():
    original_file = "jiakaoData.json"
    deduplicated_file = "jiakaoData_deduplicated.json"
    
    print("正在加载原始数据...")
    original_data = load_data(original_file)
    
    print("\n正在加载去重后数据...")
    deduplicated_data = load_data(deduplicated_file)
    
    reduction_percentage = (1 - len(deduplicated_data) / len(original_data)) * 100
    print(f"\n数据减少量: {reduction_percentage:.2f}%")
    
    # 检查重复
    duplicate_count = check_duplicates(deduplicated_data)
    
    # 分析内容完整性
    print("\n正在检查数据完整性...")
    issues = analyze_content_completeness(deduplicated_data)
    
    # 随机抽样
    sample_questions(deduplicated_data)
    
    # 总结报告
    print("\n=== 数据质量报告 ===")
    print(f"- 原始题目总数: {len(original_data)}")
    print(f"- 去重后题目数: {len(deduplicated_data)}")
    print(f"- 减少题目数量: {len(original_data) - len(deduplicated_data)} ({reduction_percentage:.2f}%)")
    print(f"- 仍有重复题目组: {duplicate_count}")
    print(f"- 数据完整性问题: {issues}")
    
    if duplicate_count == 0 and issues == 0:
        print("\n✓ 去重成功! 数据质量良好.")
    else:
        print("\n⚠ 数据存在一些问题，建议进一步优化.")

if __name__ == "__main__":
    main() 