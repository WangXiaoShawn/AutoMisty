# AutoMisty

## 🤖 AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot (IROS2025)

**论文链接**: https://arxiv.org/pdf/2503.06791  
**Presentation demo**: 

[![AutoMisty Official Presentation](https://res.cloudinary.com/marcomontalbano/image/upload/v1724291960/video_to_markdown/images/youtube--MWbNXMBj0YA-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=MWbNXMBj0YA "AutoMisty Official Presentation")

AutoMisty是一个基于多智能体大语言模型的框架，专为Misty社交机器人自动化代码生成而设计。该项目在IROS2025会议上发表，提供了完整的机器人交互、感知、规划和动作执行能力。

---

## 📁 项目结构

```
AutoMistyIROS2025/
├── AutoMisty.py              # 🚀 主程序入口
├── Agents/                   # 🧠 多智能体模块
│   ├── MistyActionAgent.py   # 动作智能体
│   ├── MistyPerceptionAgent.py # 感知智能体
│   ├── MistyPlanAgent.py     # 规划智能体
│   ├── MistyEventAgent.py    # 事件智能体
│   └── ...
├── code/mistyPy/             # 🎯 核心代码库
│   ├── CUBS_Misty.py         # 🔥 机器人核心类（必须保留）
│   ├── RobotCommands.py      # 🔥 基础命令类（必须保留）
│   └── [生成的代码文件]       # AutoMisty自动生成的代码
├── DB/                       # 🧲 向量数据库
│   ├── misty_action_db/      # 动作记忆数据库
│   ├── misty_perception_db/  # 感知记忆数据库
│   ├── misty_plan_db/        # 规划记忆数据库
│   └── misty_event_db/       # 事件记忆数据库
├── Mistydemo/                # 📚 论文实验代码
│   ├── SimpleTask/           # 简单任务示例
│   ├── CompoundTask/         # 复合任务示例
│   ├── ComplexTask/          # 复杂任务示例
│   └── ElementaryTask/       # 基础任务示例
├── OAI_CONFIG_LIST.json      # 🔑 API配置文件
└── requirements.txt          # 📦 依赖列表
```

---

## 🛠️ 安装配置

### 1. 环境要求

- **Python**: 3.8+
- **操作系统**: macOS（⚠️ 推荐使用macOS，因为项目使用视频流在本地运行，请不要在服务器上运行或尝试路由到本地）
- **硬件**: Misty II 机器人

### 2. 克隆项目

```bash
git clone <repository-url>
cd AutoMistyIROS2025
```

### 3. 安装依赖

推荐使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 安装所有依赖
pip install -r requirements.txt
```

### 4. 配置 API 密钥和 Misty IP

编辑 `OAI_CONFIG_LIST.json` 文件：

```json
[
    {
        "model": "gpt-4o",
        "api_key": "YOUR_OPENAI_API_KEY_HERE",
        "misty_ip": "YOUR_MISTY_ROBOT_IP_HERE"
    }
]
```

**重要提示**：
- 将 `YOUR_OPENAI_API_KEY_HERE` 替换为您的OpenAI API密钥
- 将 `YOUR_MISTY_ROBOT_IP_HERE` 替换为您的Misty机器人IP地址

---

## 🚀 快速开始

### 运行主程序

```bash
python AutoMisty.py
```

### 使用指南

1. **启动程序后**，请遵循代码给出的提示来运行

2. **MEM系统**：对于您认为有帮助的代码，请输入`MEM`，系统会自动将其存入本地数据库，在下次有相同任务时会自动检索相关代码

3. **AV流交互**：当有音视频流与您交互时，测试完毕请使用 `Ctrl+X` 退出AV界面，以维持与AutoMisty的持续交互

4. **代码生成**：AutoMisty生成的所有代码都保存在 `code/mistyPy/` 文件夹内

---

## 🔥 核心组件（禁止删除）

### CUBS_Misty.py 和 RobotCommands.py

⚠️ **重要警告**：AutoMisty生成的所有代码都会继承 `CUBS_Misty.py` 中的 `Robot` 类，因此**请不要删除**以下核心文件：

- **`CUBS_Misty.py`**: 机器人核心功能类
- **`RobotCommands.py`**: 基础API命令类

这两个文件是整个系统的基础架构，删除将导致所有生成的代码无法运行。

---

## 📄 引用

如果您在研究中使用了AutoMisty，请引用以下论文：

```bibtex
@article{wang2025automisty,
  title={AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot},
  author={Wang, Xiao and Dong, Lu and Rangasrinivasan, Sahana and Nwogu, Ifeoma and Setlur, Srirangaraj and Govindaraju, Venugopal},
  journal={arXiv preprint arXiv:2503.06791},
  year={2025}
}
```
