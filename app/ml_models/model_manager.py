import os
import json

MODEL_REGISTRY_FILE = "model_registry.json"

class ModelManager:
    def __init__(self, base_dir="app/ml_models"):
        self.base_dir = base_dir
        self.registry = self._load_registry()
        self.models = {}  # 缓存加载的模型

    def _load_registry(self):
        """加载模型注册表"""
        if os.path.exists(MODEL_REGISTRY_FILE):
            with open(MODEL_REGISTRY_FILE, "r") as file:
                return json.load(file)
        return {}

    def _save_registry(self):
        """保存模型注册表"""
        with open(MODEL_REGISTRY_FILE, "w") as file:
            json.dump(self.registry, file, indent=4)

    def scan_models(self):
        """扫描模型文件夹并更新注册表"""
        model_folders = [f for f in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, f))]
        for folder in model_folders:
            model_path = os.path.join(self.base_dir, folder, "models")
            if os.path.exists(model_path):
                self.registry[folder] = {
                    "path": model_path,
                    "type": folder  # 将文件夹名作为模型类型
                }
        self._save_registry()

    def list_registered_models(self):
        """列出已注册的模型"""
        return list(self.registry.keys())

    def load_model(self, model_name):
        """加载指定模型"""
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' is not registered.")
        if model_name in self.models:
            return self.models[model_name]  # 如果模型已加载，直接返回

        model_info = self.registry[model_name]
        model_path = model_info["path"]

        # 根据模型类型加载不同模型
        if model_info["type"] == "question_generation":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        elif model_info["type"] == "answer_generation":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        elif model_info["type"] == "distractor_generation":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_info['type']}")

        self.models[model_name] = (model, tokenizer)
        return model, tokenizer
