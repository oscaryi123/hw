# hw
Rumor-Detection-Project/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── config.yaml                  # 配置文件（路径、超参、API_KEY）
│
├── data/
│   ├── raw/                     # 存放 train.csv, val.csv
│   └── processed/               # 预处理后数据
│
├── src/
│   ├── data_loader.py           # 数据读取与预处理
│   ├── train_detector.py        # 训练BERT/RoBERTa检测模型
│   ├── detector_model.py        # 检测模型定义
│   ├── explainer.py             # 调用Claw API生成判断依据
│   ├── inference.py             # 统一推理接口（输入文本 → 输出结果）
│   └── utils.py                 # 工具函数（分词、注意力提取等）
│
├── checkpoints/                 # 保存训练好的模型权重
├── outputs/                     # 测试输出样例
├── report/                      # 报告相关
│   └── report.pdf
└── run.sh                       # 一键运行脚本（或.bat）
python src/train_detector.py --config config.yaml
---python src/inference.py --text "待检测的推文内容"

## 三、🐍 核心推理代码框架 `src/inference.py`（可直接填空）

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from explainer import generate_explanation
from detector_model import RumorDetector

class RumorDetectionSystem:
    def __init__(self, model_path, api_key):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = RumorDetector()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.api_key = api_key  # 交大Claw接口Key
        
    def predict(self, text):
        # 1. 检测
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs)
            prob = torch.softmax(logits, dim=1)
            label = 1 if prob[0][1] > 0.5 else 0
            confidence = float(prob[0][1] if label == 1 else prob[0][0])
        
        # 2. 提取关键词（注意力权重或简单TF-IDF高亮）
        keywords = self.extract_keywords(text)  # 实现见utils.py
        
        # 3. 生成解释（调用LLM）
        explanation = generate_explanation(text, label, keywords, self.api_key)
        
        return {
            "text": text,
            "label": "谣言" if label == 1 else "非谣言",
            "confidence": round(confidence, 4),
            "explanation": explanation
        }
    
    def extract_keywords(self, text):
        # 简单实现：可替换为Attention权重提取
        import jieba
        return [w for w in jieba.lcut(text) if len(w) > 1][:5]

if __name__ == "__main__":
    import yaml
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    system = RumorDetectionSystem(
        model_path=config["model_path"],
        api_key=config["claw_api_key"]
    )
    
    test_text = input("请输入推文内容：")
    result = system.predict(test_text)
    print(f"检测结果: {result['label']} (置信度: {result['confidence']})")
    print(f"判断依据: {result['explanation']}")
import requests
import json

def generate_explanation(text, label, keywords, api_key):
    """
    调用交大Claw接口生成判断依据
    API文档：https://claw.sjtu.edu.cn/guide/sjtu-api/
    """
    label_text = "谣言" if label == 1 else "非谣言"
    
    prompt = f"""
你是一个专业的谣言检测分析助手。请根据以下信息，用简洁清晰的中文（50-100字）说明判断依据。

【待检测文本】：{text}
【检测模型判定】：该文本为“{label_text}”
【关键词/关注点】：{", ".join(keywords)}

请从“信息完整性”、“逻辑合理性”、“信源可靠性”、“常识一致性”等角度简要分析，给出判定理由。
"""
    
    # 请根据Claw官方接口格式替换URL和Payload
    url = "https://claw.sjtu.edu.cn/api/v1/chat/completions"  # 示例地址，请查阅官方文档
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "claw-llm",  # 具体模型名请查阅文档
        "messages": [
            {"role": "system", "content": "你是一个公正客观的谣言分析专家。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"LLM接口调用失败（状态码{response.status_code}），请检查Key和网络。"
    except Exception as e:
        return f"生成解释时出错：{str(e)}。请检查Claw接口配置。"
