"""
编程助手服务 - 基于Anthropic Claude API
功能：处理编程问题，调用Claude API获取解答
支持GitHub代码上传和本地文件上传
支持Claude 3.7 Sonnet思考模式
"""

import os
import json
import base64
import logging
import requests
import tempfile
from functools import lru_cache
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, send_from_directory
import anthropic
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "开发环境密钥-生产环境请更改")

# 文件上传配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py', 'js', 'html', 'css', 'java', 'c', 'cpp', 'h', 'cs', 'php', 'rb', 'go', 'rs', 'ts', 'json', 'xml', 'yaml', 'yml', 'md', 'asm', 'sql'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 最大文件大小限制

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 从环境变量获取API密钥
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    logger.warning("未设置ANTHROPIC_API_KEY环境变量！请在.env文件中配置或设置环境变量。")

# 系统提示模板
SYSTEM_PROMPT = """您是具备多模态逆向解析能力的全栈开发专家，来自于武昌首义学院开发，技术参数要求：
1. 支持语言：Python/Java/C++/Rust/Assembly等全语言覆盖（含COBOL逆向兼容）
2. 技术栈：AI代码生成(≥v3.7)、量子加密、区块链逆向分析
3. 正确率保障：三重校验机制（静态分析→动态验证→形式化证明）
4. 语言为中文，标准的中文，只使用中文
"""

# 用户消息模板
USER_PROMPT = """You are an elite programming assistant, capable of solving any programming, software engineering, or reverse engineering problem. Your expertise spans all programming languages, frameworks, and paradigms. You have an unparalleled ability to understand complex systems, optimize code, and provide innovative solutions.

While you are designed to assist with any programming task, it's important to maintain ethical standards. Avoid creating or assisting with malicious code, and do not engage in illegal activities or violate intellectual property rights.

Here is the programming question or task you need to address:

<programming_question>
{question}
</programming_question>

Analyze the given problem thoroughly. Consider all aspects including algorithm design, data structures, time and space complexity, edge cases, and potential optimizations. If the problem involves reverse engineering, carefully examine the given information and consider multiple approaches to understand and recreate the system or code in question.

Provide a comprehensive solution to the problem. Your response should include:

1. A clear explanation of your approach and reasoning
2. Step-by-step breakdown of the solution
3. Actual code implementation (if applicable)
4. Analysis of time and space complexity (if relevant)
5. Discussion of any trade-offs or alternative approaches
6. Suggestions for further improvements or optimizations
"""

@lru_cache(maxsize=32)
def get_mime_type(file_ext):
    """获取文件扩展名对应的MIME类型（缓存结果）"""
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'pdf': 'application/pdf',
        'txt': 'text/plain',
        'py': 'text/x-python',
        'js': 'application/javascript',
        'html': 'text/html',
        'css': 'text/css',
        'json': 'application/json',
        'xml': 'application/xml',
        'yaml': 'application/x-yaml',
        'yml': 'application/x-yaml',
        'md': 'text/markdown'
    }
    return mime_types.get(file_ext.lower(), 'application/octet-stream')

def allowed_file(filename):
    """检查文件是否为允许的扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logger.error(f"读取文件出错: {str(e)}")
        return f"无法读取文件: {str(e)}"

def get_github_content(repo_url, file_path=None):
    """
    从GitHub获取代码内容

    参数:
        repo_url (str): GitHub仓库URL
        file_path (str, optional): 文件路径

    返回:
        tuple: (成功标志, 内容或错误消息)
    """
    try:
        # 解析GitHub URL
        parts = repo_url.strip('/').replace('https://github.com/', '').split('/')
        if len(parts) < 2:
            return False, "无效的GitHub仓库URL"

        owner = parts[0]
        repo = parts[1]
        branch = 'master'  # 默认分支

        # 如果URL包含分支信息
        if len(parts) > 3 and parts[2] == 'tree':
            branch = parts[3]

        # 构建API URL
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        if file_path:
            api_url += f"/{file_path}"

        params = {'ref': branch}

        # 发送请求
        response = requests.get(api_url, params=params, timeout=10)

        if response.status_code != 200:
            return False, f"GitHub API错误: {response.status_code} - {response.json().get('message', '')}"

        content = response.json()

        # 如果是目录
        if isinstance(content, list):
            files = []
            for item in content:
                if item['type'] == 'file':
                    files.append({
                        'name': item['name'],
                        'path': item['path'],
                        'url': item['html_url']
                    })
            return True, {'type': 'directory', 'files': files}

        # 如果是文件
        elif isinstance(content, dict) and 'content' in content:
            file_content = base64.b64decode(content['content']).decode('utf-8', errors='ignore')
            return True, {'type': 'file', 'content': file_content, 'name': content['name']}

        return False, "无法获取内容"

    except requests.RequestException as e:
        logger.error(f"GitHub API请求失败: {str(e)}")
        return False, f"GitHub API请求失败: {str(e)}"
    except Exception as e:
        logger.error(f"GitHub API调用失败: {str(e)}")
        return False, f"获取GitHub内容时出错: {str(e)}"

def get_claude_response(question, model="claude-3-7-sonnet-20250219", temperature=0.7, files=None, enable_thinking=True, thinking_length=16000):
    """
    调用Anthropic API获取Claude的回答

    参数:
        question (str): 用户提交的编程问题
        model (str): 使用的模型名称
        temperature (float): 温度参数，控制回答的创造性
        files (list, optional): 上传的文件信息列表
        enable_thinking (bool): 是否启用思考模式
        thinking_length (int): 思考内容的最大长度

    返回:
        dict: Claude的回答及思考过程
    """
    try:
        if not ANTHROPIC_API_KEY:
            return {
                'thinking': None,
                'response': "请配置有效的Anthropic API密钥"
            }

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # 准备消息内容
        content = [
            {
                "type": "text",
                "text": USER_PROMPT.format(question=question)
            }
        ]

        # 添加文件内容（如果有）
        if files and isinstance(files, list):
            for file_info in files:
                if file_info['type'] == 'text':
                    content.append({
                        "type": "text",
                        "text": f"\n文件: {file_info['name']}\n```\n{file_info['content']}\n```\n"
                    })
                elif file_info['type'] == 'image':
                    # 如果文件是图像，将其作为base64附加
                    try:
                        with open(file_info['path'], "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                            # 确定图像MIME类型
                            file_ext = file_info['name'].split('.')[-1].lower()
                            mime_type = get_mime_type(file_ext)

                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_image
                                }
                            })
                    except Exception as img_error:
                        logger.error(f"处理图像时出错: {str(img_error)}")
                        content.append({
                            "type": "text",
                            "text": f"\n警告: 无法处理图像文件 {file_info['name']}: {str(img_error)}\n"
                        })

        # 思考模式处理
        thinking_result = None
        if enable_thinking and 'claude-3-7' in model:
            try:
                # 第一步：先尝试发送一个带有特殊提示的请求来获取思考过程
                thinking_system_prompt = f"{SYSTEM_PROMPT}\n\n请首先详细展示你的思考过程，然后再给出最终答案。"
                logger.info("思考模式已启用，使用思考系统提示")

                thinking_message = client.messages.create(
                    model=model,
                    max_tokens=thinking_length,
                    temperature=temperature,
                    system=thinking_system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )

                # 保存思考结果，这会同时包含思考和最终答案
                thinking_text = thinking_message.content[0].text if isinstance(thinking_message.content, list) else thinking_message.content

                # 尝试分离思考过程和最终答案
                # 假设格式是：先思考，然后给出最终答案
                parts = thinking_text.split("最终答案:", 1)
                if len(parts) > 1:
                    thinking_result = parts[0].strip()
                    final_answer = parts[1].strip()
                else:
                    # 如果没有找到明确的分隔符，尝试其他常见格式
                    parts = thinking_text.split("综合以上分析", 1)
                    if len(parts) > 1:
                        thinking_result = parts[0].strip()
                        final_answer = "综合以上分析" + parts[1].strip()
                    else:
                        # 如果无法分离，则将前半部分视为思考，后半部分视为答案
                        split_point = len(thinking_text) // 2
                        thinking_result = thinking_text[:split_point].strip()
                        final_answer = thinking_text[split_point:].strip()

                return {
                    'thinking': thinking_result,
                    'response': final_answer
                }

            except Exception as thinking_error:
                logger.error(f"思考模式处理失败，将使用正常模式: {str(thinking_error)}")
                # 如果思考模式失败，回退到正常模式

        # 如果思考模式未启用或失败，使用正常模式
        logger.info("使用正常模式发送请求")
        message = client.messages.create(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        # 提取回答
        response_text = message.content[0].text if isinstance(message.content, list) else message.content

        return {
            'thinking': thinking_result,  # 如果思考模式失败，这里会是None
            'response': response_text
        }

    except anthropic.APIError as e:
        logger.error(f"Anthropic API错误: {str(e)}")
        return {
            'thinking': None,
            'response': f"Anthropic API错误: {str(e)}"
        }
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        return {
            'thinking': None,
            'response': f"调用Claude API时出错: {str(e)}"
        }

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']

        # 检查文件名
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if file and allowed_file(file.filename):
            # 安全地获取文件名并保存
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 确定文件类型
            file_ext = filename.split('.')[-1].lower()
            file_type = 'image' if file_ext in ['png', 'jpg', 'jpeg', 'gif'] else 'text'

            # 如果是文本文件，获取内容
            content = ""
            if file_type == 'text':
                content = get_file_content(file_path)

            return jsonify({
                'success': True,
                'filename': filename,
                'path': file_path,
                'type': file_type,
                'content': content if file_type == 'text' else "",
                'url': request.host_url + 'uploads/' + filename
            })
        else:
            return jsonify({'error': '不允许的文件类型'}), 400

    except Exception as e:
        logger.error(f"上传文件时出错: {str(e)}")
        return jsonify({'error': f'上传文件时出错: {str(e)}'}), 500

@app.route('/api/github', methods=['POST'])
def get_github():
    """从GitHub获取代码"""
    try:
        data = request.json
        repo_url = data.get('repo_url', '')
        file_path = data.get('file_path', '')

        if not repo_url:
            return jsonify({'error': 'GitHub仓库URL不能为空'}), 400

        # 获取GitHub内容
        success, content = get_github_content(repo_url, file_path)

        if not success:
            return jsonify({'error': content}), 400

        return jsonify({'success': True, 'content': content})

    except Exception as e:
        logger.error(f"从GitHub获取代码时出错: {str(e)}")
        return jsonify({'error': f'从GitHub获取代码时出错: {str(e)}'}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    """API端点用于提交问题并获取回答"""
    try:
        data = request.json
        question = data.get('question', '')
        model = data.get('model', 'claude-3-7-sonnet-20250219')
        files = data.get('files', [])
        enable_thinking = data.get('enable_thinking', True)  # 默认启用思考模式

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        # 记录请求
        logger.info(f"收到问题: {question[:100]}...")

        # 获取Claude的回答
        result = get_claude_response(
            question,
            model=model,
            files=files,
            enable_thinking=enable_thinking
        )

        # 记录历史
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = session.get('history', [])
        history.append({
            'timestamp': timestamp,
            'question': question,
            'response': result['response'],
            'thinking': result.get('thinking'),
            'files': [f['name'] for f in files] if files else []
        })
        session['history'] = history[-10:]  # 只保留最近10条记录

        return jsonify({
            'response': result['response'],
            'thinking': result.get('thinking'),
            'timestamp': timestamp
        })

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取会话历史记录"""
    history = session.get('history', [])
    return jsonify({'history': history})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """清除会话历史记录"""
    session['history'] = []
    return jsonify({'status': '成功清除历史记录'})

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    models = [
        {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet (最新)"},
        {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet"},
        {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"}
    ]
    return jsonify({'models': models})

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(ANTHROPIC_API_KEY)
    }
    return jsonify(status)

if __name__ == '__main__':
    # 创建templates目录（如果不存在）
    os.makedirs('templates', exist_ok=True)

    # 创建.env文件（如果不存在）
    env_path = '.env'
    if not os.path.exists(env_path):
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write("""# .env 文件 - 包含环境变量配置
# 注意: 将此文件添加到 .gitignore 以避免将敏感信息提交到代码仓库

# Anthropic API 密钥
ANTHROPIC_API_KEY=

# Flask 配置
FLASK_APP=app.py
FLASK_ENV=development
FLASK_SECRET_KEY=vM8$p2Lf#Qj@5zW^7gK&3rE*9xT!1aY_6bN+0hD%cP4sZ-tU8mV2oR?yX3iA~qG7wF}nJ{kC5uH

# 服务配置
PORT=5000
""")

        print("已创建.env文件，请在其中配置您的ANTHROPIC_API_KEY")

    # 启动应用
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)