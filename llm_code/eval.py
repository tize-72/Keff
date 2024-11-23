import re
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz 
import json
def string_match(pattern, text, method='contains', threshold=0.6):
    """
    判断字符串pattern是否在text中出现
    
    参数:
        pattern: 要搜索的字符串
        text: 被搜索的文本
        method: 匹配方法 ('contains', 'fuzzy', 'regex', 'sequence')
        threshold: 模糊匹配的阈值 (0-1)
    
    返回:
        bool: 是否匹配
    """
    # 转换为小写以实现不区分大小写
    pattern = pattern.lower()
    text = text.lower()
    
    if method == 'contains':
        # 简单包含匹配
        return pattern in text
    
    elif method == 'fuzzy':
        # 使用fuzzywuzzy进行模糊匹配
        # 需要先安装fuzzywuzzy: pip install fuzzywuzzy
        ratio = fuzz.partial_ratio(pattern, text)
        return ratio / 100 >= threshold
    
    elif method == 'regex':
        # 使用正则表达式匹配
        try:
            pattern = '.*?'.join(map(re.escape, pattern))
            return bool(re.search(pattern, text))
        except:
            return False
    
    elif method == 'sequence':
        # 使用序列匹配
        matcher = SequenceMatcher(None, pattern, text)
        return matcher.ratio() >= threshold
    
    else:
        raise ValueError("Unsupported method")

if __name__ == '__main__':
    file_path = './mistral_result.json'
    # 判断结果是否包含
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        true = 0 
        false = 0
        # 遍历列表中的每一项
        for index, item in enumerate(data):
            # 这里处理每一项
            print(f"第{index+1}项:", item)
            flag = string_match(item['true'], item['llm_response'], method='contains')
            print(flag)
            if flag:
                true += 1
            else:
                false += 1
        f1_rate = round(true/len(data),4)
        print(f"最终的准确率为{f1_rate}")
    except FileNotFoundError:
        print(f"找不到文件: {file_path}")
    except json.JSONDecodeError:
        print("JSON格式错误")
    except Exception as e:
        print(f"发生错误: {str(e)}")