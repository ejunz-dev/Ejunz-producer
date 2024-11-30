from transformers import Trainer, TrainingArguments
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import os
from datasets import load_dataset
from tempfile import NamedTemporaryFile
import torch

class ModelTrainer:
    def fine_tune_model(self, model_path, dataset, epochs):
        # 加载模型和分词器
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)

        # 如果 dataset 是列表（直接数据），创建一个临时文件来存储数据
        if isinstance(dataset, list):
            with NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as temp_file:
                json.dump(dataset, temp_file)  # 使用 json.dump 保存数据
                dataset_path = temp_file.name
        elif isinstance(dataset, str) and os.path.exists(dataset):
            # 如果是文件路径，直接使用
            dataset_path = dataset
        else:
            raise ValueError("Invalid dataset format. Must be a file path or a list of data.")

        # 加载数据集
        dataset = load_dataset("json", data_files=dataset_path)['train']

        # 数据预处理
        def preprocess_function(examples):
            inputs = examples['question']
            targets = examples['answer']
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = dataset.map(preprocess_function, batched=True)

        # 微调参数
        training_args = TrainingArguments(
            output_dir="app/ml_models/question_generation/output",
            evaluation_strategy="steps",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="app/ml_models/question_generation/logs",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        # 开始微调
        trainer.train()

        # 保存微调后的模型
        trainer.save_model("app/ml_models/question_generation/output")
        tokenizer.save_pretrained("app/ml_models/question_generation/output")

        # 删除临时文件（如果创建了）
        if isinstance(dataset, list):
            os.remove(dataset_path)

        return {"status": "fine-tune completed", "model_path": "app/ml_models/question_generation/output"}

    def load_model(self, model_path):
        """
        加载已经微调完成的模型
        """
        try:
            # 使用 Hugging Face 的 from_pretrained 方法加载模型和分词器
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model.eval()  # 切换到推理模式
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
