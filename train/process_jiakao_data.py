import json
import re
from collections import Counter
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

def load_data(file_path):
    """加载驾考题目数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"总题目数: {len(data)}")
    return data

def analyze_data(data):
    """基本数据分析"""
    # 检查数据结构
    print("数据结构示例:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
    
    # 统计题型
    question_types = Counter()
    for item in data:
        input_text = item["input"]
        if "题型：判断" in input_text:
            question_types["判断题"] += 1
        elif "题型：选择题" in input_text:
            question_types["选择题"] += 1
        else:
            question_types["其他"] += 1
    
    print("\n题型统计:")
    for qtype, count in question_types.items():
        print(f"{qtype}: {count}")
    
    # 检查是否有明显重复题目（完全相同的题目文本）
    question_text_count = Counter()
    for item in data:
        question_text = extract_question_text(item["input"])
        question_text_count[question_text] += 1
    
    duplicate_count = sum(1 for text, count in question_text_count.items() if count > 1)
    print(f"\n完全相同的题目数量: {duplicate_count}")
    
    # 显示部分重复最多的题目
    print("\n重复最多的题目:")
    for text, count in question_text_count.most_common(5):
        print(f"- 题目: {text[:50]}... (重复 {count} 次)")

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

def find_similar_questions(data, threshold=0.85):
    """查找相似题目"""
    # 提取题目文本
    questions = []
    question_hashes = []
    for item in data:
        question_text = extract_question_text(item["input"])
        questions.append(question_text)
        question_hashes.append(calculate_text_hash(question_text))
    
    # 先通过哈希值查找完全相同的题目
    hash_to_indices = {}
    exact_duplicates = []
    for i, hash_value in enumerate(question_hashes):
        if hash_value in hash_to_indices:
            for j in hash_to_indices[hash_value]:
                exact_duplicates.append((j, i, 1.0))
            hash_to_indices[hash_value].append(i)
        else:
            hash_to_indices[hash_value] = [i]
    
    print(f"\n发现 {len(exact_duplicates)} 对完全相同的题目")
    
    # 使用TF-IDF向量化找出相似但不完全相同的题目
    print("正在使用TF-IDF分析相似题目...")
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: jieba.lcut(x))
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    
    # 计算余弦相似度
    similar_pairs = []
    batch_size = 1000  # 批处理大小，避免内存溢出
    
    for i in range(0, len(questions), batch_size):
        batch_end = min(i + batch_size, len(questions))
        batch_matrix = tfidf_matrix[i:batch_end]
        
        cosine_sim = cosine_similarity(batch_matrix, tfidf_matrix)
        
        for j_offset, similarities in enumerate(cosine_sim):
            j = i + j_offset
            # 只考虑后面的索引，避免重复
            for k, sim in enumerate(similarities[j+1:], j+1):
                if sim > threshold and sim < 1.0:  # 排除完全相同的题目
                    similar_pairs.append((j, k, sim))
    
    # 合并完全相同和相似的题目对
    all_pairs = exact_duplicates + similar_pairs
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n发现 {len(similar_pairs)} 对相似题目 (阈值: {threshold})")
    print(f"总计 {len(all_pairs)} 对需要处理的题目对")
    
    # 输出前10对相似题目
    for idx, (i, j, sim) in enumerate(all_pairs[:10]):
        print(f"\n相似题目对 {idx+1} (相似度: {sim:.4f}):")
        print(f"题目A: {questions[i]}")
        print(f"题目B: {questions[j]}")
    
    return all_pairs

def evaluate_question_quality(item):
    """评估题目的质量"""
    # 这里简单实现，根据题目的长度、解释的详细程度等进行评分
    score = 0
    
    question_text = extract_question_text(item["input"])
    explanation = item["output"]
    
    # 1. 题目长度适中（不太短也不太长）
    q_len = len(question_text)
    if 10 <= q_len <= 100:
        score += 3
    elif q_len > 100:
        score += 2
    else:
        score += 1
    
    # 2. 解释详细程度
    if len(explanation) > 200:
        score += 3
    elif len(explanation) > 100:
        score += 2
    else:
        score += 1
    
    # 3. 解释是否包含法规引用
    if "《" in explanation and "》" in explanation:
        score += 2
    
    # 4. 是否包含"因此答案选择"等结论性表述
    if "因此" in explanation and "选择" in explanation:
        score += 2
        
    return score

def deduplicate_questions(data, similar_pairs, output_file, similarity_threshold=0.95):
    """去重处理"""
    # 计算每个题目的质量分数
    quality_scores = [evaluate_question_quality(item) for item in data]
    
    # 构建图表示重复关系
    groups = {}  # 将相似题目分组
    
    for i, j, sim in similar_pairs:
        if sim < similarity_threshold:
            continue
            
        # 将i和j放入同一组
        found = False
        for group_id, members in groups.items():
            if i in members or j in members:
                members.add(i)
                members.add(j)
                found = True
                break
                
        if not found:
            # 创建新组
            group_id = len(groups)
            groups[group_id] = {i, j}
    
    # 从每组中选择质量最高的题目
    to_keep = set(range(len(data)))  # 默认保留所有题目
    
    for group_id, members in groups.items():
        if len(members) <= 1:
            continue
            
        # 按质量分数排序组内题目
        sorted_members = sorted(list(members), key=lambda idx: quality_scores[idx], reverse=True)
        
        # 保留最高质量的题目
        best_idx = sorted_members[0]
        
        # 其余题目从保留列表中删除
        for idx in sorted_members[1:]:
            if idx in to_keep:
                to_keep.remove(idx)
    
    # 创建精简后的数据
    deduplicated_data = [data[i] for i in to_keep]
    
    print(f"\n去重后题目数量: {len(deduplicated_data)} (删除了 {len(data) - len(deduplicated_data)} 个题目)")
    
    # 保存精简后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_data, f, ensure_ascii=False, indent=2)
    
    print(f"精简后的数据已保存到 {output_file}")

def main():
    input_file = "jiakaoData.json"
    output_file = "jiakaoData_deduplicated.json"
    
    print("正在加载数据...")
    data = load_data(input_file)
    
    print("\n正在分析数据...")
    analyze_data(data)
    
    print("\n正在查找相似题目...")
    similar_pairs = find_similar_questions(data, threshold=0.85)
    
    print("\n正在去重...")
    deduplicate_questions(data, similar_pairs, output_file, similarity_threshold=0.92)

if __name__ == "__main__":
    main() 