"""
编程助手服务 - 基于Anthropic Claude API
功能：处理编程问题，调用Claude API获取解答
支持GitHub代码上传和本地文件上传
"""

import os
import json
import base64
import logging
import requests
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, send_from_directory
import anthropic

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

# 从环境变量获取API密钥，或使用默认值（仅用于开发）
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")

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
        response = requests.get(api_url, params=params)

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

    except Exception as e:
        logger.error(f"GitHub API调用失败: {str(e)}")
        return False, f"获取GitHub内容时出错: {str(e)}"

def get_claude_response(question, model="claude-3-7-sonnet-20250219", temperature=0.7, files=None):
    """
    调用Anthropic API获取Claude的回答

    参数:
        question (str): 用户提交的编程问题
        model (str): 使用的模型名称
        temperature (float): 温度参数，控制回答的创造性
        files (list, optional): 上传的文件信息列表

    返回:
        str: Claude的回答
    """
    try:
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
                            mime_type = {
                                'png': 'image/png',
                                'jpg': 'image/jpeg',
                                'jpeg': 'image/jpeg',
                                'gif': 'image/gif'
                            }.get(file_ext, 'image/jpeg')

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

        message = client.messages.create(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": content}] if isinstance(content, str) else content
                }
            ]
        )

        return message.content[0].text if isinstance(message.content, list) else message.content

    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        return f"调用Claude API时出错: {str(e)}"

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

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        # 记录请求
        logger.info(f"收到问题: {question[:100]}...")

        # 获取Claude的回答
        response = get_claude_response(question, model=model, files=files)

        # 记录历史
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = session.get('history', [])
        history.append({
            'timestamp': timestamp,
            'question': question,
            'response': response,
            'files': [f['name'] for f in files] if files else []
        })
        session['history'] = history[-10:]  # 只保留最近10条记录

        return jsonify({
            'response': response,
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

if __name__ == '__main__':
    # 创建templates目录（如果不存在）
    os.makedirs('templates', exist_ok=True)

    # 创建index.html文件
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>编程助手服务</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <style>
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin-top: 30px;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .response-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            white-space: pre-wrap;
            max-height: 600px;
            overflow-y: auto;
        }
        .code-block {
            background-color: #272822;
            color: #f8f8f2;
            border-radius: 5px;
            padding: 10px;
            overflow-x: auto;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0, 0, 0, 0.3);
            border-radius: 50%;
            border-top-color: #007bff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .history-item {
            cursor: pointer;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .history-item:hover {
            background-color: #f1f1f1;
        }
        .file-list {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            max-height: 200px;
            overflow-y: auto;
        }
        .file-item {
            display: flex;
            align-items: center;
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
        .file-item:last-child {
            border-bottom: none;
        }
        .file-item i {
            margin-right: 8px;
            font-size: 1.2em;
        }
        .file-item .file-name {
            flex-grow: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .file-item .file-remove {
            cursor: pointer;
            color: #dc3545;
        }
        .tab-content {
            padding: 15px;
            border: 1px solid #dee2e6;
            border-top: 0;
            border-radius: 0 0 5px 5px;
        }
        .nav-tabs .nav-link {
            font-size: 0.9rem;
        }
        .github-preview {
            max-height: 300px;
            overflow-y: auto;
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 10px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">编程助手服务</h1>
        
        <div class="row">
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">提交问题</h5>
                        <div class="form-group mb-0">
                            <select id="modelSelect" class="form-select form-select-sm">
                                <option value="claude-3-7-sonnet-20250219">Claude 3.7 Sonnet (最新)</option>
                                <option value="claude-3-5-sonnet-20240620">Claude 3.5 Sonnet</option>
                                <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                            </select>
                        </div>
                    </div>
                    <div class="card-body">
                        <form id="questionForm">
                            <div class="form-group">
                                <textarea id="questionInput" class="form-control" rows="6" placeholder="请输入您的编程问题或逆向工程任务..."></textarea>
                            </div>
                            
                            <ul class="nav nav-tabs mt-3" id="inputTabs" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="files-tab" data-bs-toggle="tab" data-bs-target="#files" type="button" role="tab">本地文件上传</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="github-tab" data-bs-toggle="tab" data-bs-target="#github" type="button" role="tab">GitHub代码获取</button>
                                </li>
                            </ul>
                            
                            <div class="tab-content" id="inputTabsContent">
                                <div class="tab-pane fade show active" id="files" role="tabpanel">
                                    <div class="mb-3">
                                        <label for="fileInput" class="form-label">上传文件（代码或图像）</label>
                                        <input class="form-control" type="file" id="fileInput" multiple>
                                        <div class="form-text">支持的文件类型: .txt, .py, .js, .html, .css, .java, .c, .cpp, .h, .cs, .php, .rb, .go, .rs, .ts, .json, .xml, .yaml, .yml, .md, .asm, .sql, .png, .jpg, .jpeg, .gif, .pdf</div>
                                    </div>
                                    
                                    <div id="fileList" class="file-list d-none">
                                        <div class="text-center text-muted">没有上传的文件</div>
                                    </div>
                                </div>
                                
                                <div class="tab-pane fade" id="github" role="tabpanel">
                                    <div class="mb-3">
                                        <label for="githubUrlInput" class="form-label">GitHub仓库URL</label>
                                        <input type="text" class="form-control" id="githubUrlInput" placeholder="https://github.com/用户名/仓库名">
                                    </div>
                                    <div class="mb-3">
                                        <label for="githubPathInput" class="form-label">文件路径（可选）</label>
                                        <input type="text" class="form-control" id="githubPathInput" placeholder="例如: src/main.py">
                                    </div>
                                    <button type="button" class="btn btn-secondary" id="fetchGithubBtn">获取代码</button>
                                    
                                    <div id="githubPreviewContainer" class="mt-3 d-none">
                                        <h6>预览:</h6>
                                        <div id="githubPreview" class="github-preview"></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="d-flex justify-content-between mt-3">
                                <button type="submit" class="btn btn-primary" id="submitBtn">
                                    <span id="loadingIndicator" class="loading d-none"></span>
                                    提交问题
                                </button>
                                <button type="button" class="btn btn-outline-secondary" id="clearBtn">清除</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Claude回答</h5>
                    </div>
                    <div class="card-body">
                        <div id="responseContainer" class="response-container">
                            <p class="text-muted text-center">您的回答将显示在这里</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">历史记录</h5>
                        <button id="clearHistoryBtn" class="btn btn-sm btn-outline-danger">清空</button>
                    </div>
                    <div class="card-body p-0">
                        <div id="historyContainer">
                            <p class="text-muted text-center p-3">暂无历史记录</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.0.0/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@2.3.6/dist/purify.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const questionForm = document.getElementById('questionForm');
            const questionInput = document.getElementById('questionInput');
            const responseContainer = document.getElementById('responseContainer');
            const loadingIndicator = document.getElementById('loadingIndicator');
            const submitBtn = document.getElementById('submitBtn');
            const clearBtn = document.getElementById('clearBtn');
            const clearHistoryBtn = document.getElementById('clearHistoryBtn');
            const historyContainer = document.getElementById('historyContainer');
            const modelSelect = document.getElementById('modelSelect');
            const fileInput = document.getElementById('fileInput');
            const fileList = document.getElementById('fileList');
            const githubUrlInput = document.getElementById('githubUrlInput');
            const githubPathInput = document.getElementById('githubPathInput');
            const fetchGithubBtn = document.getElementById('fetchGithubBtn');
            const githubPreviewContainer = document.getElementById('githubPreviewContainer');
            const githubPreview = document.getElementById('githubPreview');
            
            // 存储上传的文件
            let uploadedFiles = [];
            // 存储GitHub代码内容
            let githubContent = null;
            
            // 加载历史记录
            loadHistory();
            
            // 处理文件上传
            fileInput.addEventListener('change', async function(e) {
                const files = e.target.files;
                
                if (files.length === 0) return;
                
                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/api/upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            uploadedFiles.push(data);
                            updateFileList();
                        } else {
                            alert(`文件 ${file.name} 上传失败: ${data.error}`);
                        }
                    } catch (error) {
                        alert(`文件 ${file.name} 上传过程出错: ${error.message}`);
                    }
                }
                
                // 清空文件输入框，允许重复上传相同的文件
                fileInput.value = '';
            });
            
            // 更新文件列表显示
            function updateFileList() {
                if (uploadedFiles.length === 0) {
                    fileList.classList.add('d-none');
                    return;
                }
                
                fileList.classList.remove('d-none');
                fileList.innerHTML = '';
                
                uploadedFiles.forEach((file, index) => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    
                    const fileTypeIcon = file.type === 'image' 
                        ? '<i class="bi bi-file-image text-primary"></i>' 
                        : '<i class="bi bi-file-code text-success"></i>';
                        
                    fileItem.innerHTML = `
                        ${fileTypeIcon}
                        <div class="file-name">${file.filename}</div>
                        <div class="file-remove" data-index="${index}"><i class="bi bi-x-circle"></i></div>
                    `;
                    
                    fileList.appendChild(fileItem);
                });
                
                // 添加文件删除事件
                document.querySelectorAll('.file-remove').forEach(elem => {
                    elem.addEventListener('click', function() {
                        const index = parseInt(this.getAttribute('data-index'));
                        uploadedFiles.splice(index, 1);
                        updateFileList();
                    });
                });
            }
            
            // 从GitHub获取代码
            fetchGithubBtn.addEventListener('click', async function() {
                const repoUrl = githubUrlInput.value.trim();
                const filePath = githubPathInput.value.trim();
                
                if (!repoUrl) {
                    alert('请输入GitHub仓库URL');
                    return;
                }
                
                fetchGithubBtn.disabled = true;
                fetchGithubBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 加载中...';
                
                try {
                    const response = await fetch('/api/github', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            repo_url: repoUrl,
                            file_path: filePath
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        githubContent = data.content;
                        
                        // 显示预览
                        githubPreviewContainer.classList.remove('d-none');
                        
                        if (githubContent.type === 'file') {
                            // 显示文件内容预览
                            githubPreview.textContent = githubContent.content;
                            
                            // 添加文件内容到问题末尾
                            const fileInfo = `\n\nGitHub文件 (${githubContent.name}):\n\`\`\`\n${githubContent.content}\n\`\`\``;
                            if (!questionInput.value.includes(fileInfo)) {
                                questionInput.value += fileInfo;
                            }
                        } else if (githubContent.type === 'directory') {
                            // 显示目录文件列表
                            const filesList = githubContent.files.map(file => `- ${file.name}: ${file.url}`).join('\\n');
                            githubPreview.textContent = `目录内容:\n${filesList}`;
                            
                            // 添加文件列表到问题
                            const dirInfo = `\n\nGitHub目录文件列表:\n${filesList}`;
                            if (!questionInput.value.includes(dirInfo)) {
                                questionInput.value += dirInfo;
                            }
                        }
                    } else {
                        alert(`获取GitHub内容失败: ${data.error}`);
                    }
                } catch (error) {
                    alert(`请求失败: ${error.message}`);
                } finally {
                    fetchGithubBtn.disabled = false;
                    fetchGithubBtn.innerHTML = '获取代码';
                }
            });
            
            // 提交问题
            questionForm.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const question = questionInput.value.trim();
                if (!question) {
                    alert('请输入问题');
                    return;
                }
                
                // 显示加载状态
                loadingIndicator.classList.remove('d-none');
                submitBtn.disabled = true;
                responseContainer.innerHTML = '<p class="text-center">Claude正在思考中...</p>';
                
                try {
                    // 准备文件数据
                    const files = uploadedFiles.map(file => ({
                        name: file.filename,
                        path: file.path,
                        type: file.type,
                        content: file.content || ''
                    }));
                    
                    const response = await fetch('/api/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            question: question,
                            model: modelSelect.value,
                            files: files
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // 渲染回答（使用markdown解析）
                        const cleanHtml = DOMPurify.sanitize(marked.parse(data.response));
                        responseContainer.innerHTML = cleanHtml;
                        
                        // 高亮代码块
                        document.querySelectorAll('pre code').forEach(block => {
                            if (window.hljs) {
                                hljs.highlightBlock(block);
                            }
                        });
                        
                        // 更新历史记录
                        loadHistory();
                    } else {
                        responseContainer.innerHTML = `<p class="text-danger">错误: ${data.error}</p>`;
                    }
                } catch (error) {
                    responseContainer.innerHTML = `<p class="text-danger">请求失败: ${error.message}</p>`;
                } finally {
                    // 恢复按钮状态
                    loadingIndicator.classList.add('d-none');
                    submitBtn.disabled = false;
                }
            });
            
            // 清除输入
            clearBtn.addEventListener('click', function() {
                questionInput.value = '';
                responseContainer.innerHTML = '<p class="text-muted text-center">您的回答将显示在这里</p>';
                uploadedFiles = [];
                updateFileList();
                githubContent = null;
                githubUrlInput.value = '';
                githubPathInput.value = '';
                githubPreviewContainer.classList.add('d-none');
            });
            
            // 清除历史记录
            clearHistoryBtn.addEventListener('click', async function() {
                if (confirm('确定要清空所有历史记录吗？')) {
                    try {
                        const response = await fetch('/api/clear_history', {
                            method: 'POST'
                        });
                        
                        if (response.ok) {
                            historyContainer.innerHTML = '<p class="text-muted text-center p-3">暂无历史记录</p>';
                        }
                    } catch (error) {
                        alert(`清除历史记录失败: ${error.message}`);
                    }
                }
            });
            
            // 加载历史记录
            async function loadHistory() {
                try {
                    const response = await fetch('/api/history');
                    const data = await response.json();
                    
                    if (response.ok && data.history.length > 0) {
                        historyContainer.innerHTML = '';
                        
                        data.history.forEach(item => {
                            const historyItem = document.createElement('div');
                            historyItem.className = 'history-item';
                            
                            let fileInfo = '';
                            if (item.files && item.files.length > 0) {
                                fileInfo = `<small class="text-info"><i class="bi bi-paperclip"></i> ${item.files.length} 个文件</small>`;
                            }
                            
                            historyItem.innerHTML = `
                                <div class="d-flex justify-content-between">
                                    <small class="text-muted">${item.timestamp}</small>
                                    ${fileInfo}
                                </div>
                                <p class="mb-0 text-truncate">${item.question.substring(0, 50)}${item.question.length > 50 ? '...' : ''}</p>
                            `;
                            
                            historyItem.addEventListener('click', function() {
                                questionInput.value = item.question;
                                const cleanHtml = DOMPurify.sanitize(marked.parse(item.response));
                                responseContainer.innerHTML = cleanHtml;
                            });
                            
                            historyContainer.appendChild(historyItem);
                        });
                    } else {
                        historyContainer.innerHTML = '<p class="text-muted text-center p-3">暂无历史记录</p>';
                    }
                } catch (error) {
                    console.error('加载历史记录失败:', error);
                }
            }
        });
    </script>
    <!-- 可选的代码高亮 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/styles/monokai.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/highlight.min.js"></script>
    <script>
        if (window.hljs) {
            hljs.highlightAll();
        }
    </script>
</body>
</html>
""")

    # 启动应用
    app.run(host='0.0.0.0', port=5000, debug=True)