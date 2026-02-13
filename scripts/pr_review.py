import os
import sys
import requests
from github import Github, Auth

# ================= 配置区域 =================
# 你可以在这里修改你想使用的模型
# 推荐: 
# "google/gemini-2.0-flash-001" (速度极快，免费/极便宜，上下文超长)
# "anthropic/claude-3.5-sonnet" (代码能力最强，但较贵)
# "deepseek/deepseek-chat" (性价比之王)
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

SYSTEM_PROMPT = """
你是一个资深的代码审查专家 (Code Review Agent)。
你的任务是审查 GitHub 的 Pull Request 代码变更。
请总结变更的内容
"""
# ===========================================

def get_pr_diff(repo, pr_number):
    """获取 PR 的 diff 内容"""
    pr = repo.get_pull(pr_number)
    
    # 获取 Diff 的标准方式
    headers = {
        'Authorization': f'token {os.getenv("GITHUB_TOKEN")}',
        'Accept': 'application/vnd.github.v3.diff' 
    }
    
    response = requests.get(pr.url, headers=headers)
    response.raise_for_status()
    return response.text

def analyze_code_with_llm(diff_content):
    """通过 OpenRouter 调用 LLM"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return "❌ 无法进行审查：未配置 LLM_API_KEY。"

    # OpenRouter 标准 API 地址
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # 截断 Diff 以防止超长 (OpenRouter 部分模型支持 1M+ context，但为了省钱还是截断一下)
    # Gemini Flash 支持 1M context，这里可以设置得很大
    max_len = 1000 
    truncated_diff = diff_content[:max_len] + ("\n...(diff truncated due to length)" if len(diff_content) > max_len else "")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"以下是 PR 的 Diff 内容：\n\n```diff\n{truncated_diff}\n```"}
        ],
        # OpenRouter 特定参数
        "temperature": 0.2,
        "top_p": 0.9,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # OpenRouter 推荐加这两个 Header 以便在后台统计
        "HTTP-Referer": "https://github.com/my-repo-agent", 
        "X-Title": "GitHub PR Review Agent"
    }

    try:
        # timeout 设置稍微长一点，防止大模型思考时间过长
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # 调试信息：如果出错，打印具体内容
        if response.status_code != 200:
            print(f"DEBUG: OpenRouter Error Status: {response.status_code}")
            print(f"DEBUG: OpenRouter Response: {response.text}")
            response.raise_for_status()

        result = response.json()
        
        # 兼容性处理：OpenRouter 有时返回 content 为 None (虽然罕见)
        content = result['choices'][0]['message'].get('content')
        if not content:
            return "❌ AI 返回了空内容，请检查 OpenRouter 日志。"
            
        return content

    except Exception as e:
        return f"🤖 LLM 调用失败: {str(e)}"

def main():
    # 从环境变量获取配置
    github_token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY") 
    pr_number = os.getenv("PR_NUMBER")

    if not all([github_token, repo_name, pr_number]):
        print("Missing environment variables (GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER).")
        sys.exit(1)

    try:
        # 使用 Auth 认证 (解决 DeprecationWarning)
        auth = Auth.Token(github_token)
        g = Github(auth=auth)
        
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(int(pr_number))

        print(f"🚀 开始审查 PR #{pr_number} : {pr.title} ...")

        # 1. 获取 Diff
        try:
            diff_text = get_pr_diff(repo, int(pr_number))
        except Exception as e:
            print(f"❌ 获取 Diff 失败: {e}")
            sys.exit(1)
        
        if not diff_text.strip():
            print("⚠️ Diff 为空，跳过审查。")
            return

        print(f"📄 Diff 获取成功 (长度: {len(diff_text)} chars)，正在发送给 OpenRouter ({OPENROUTER_MODEL})...")

        # 2. LLM 审查
        review_comment = analyze_code_with_llm(diff_text)
        
        # 3. 发布评论
        print(f"✅ 审查完成，准备提交评论...")
        
        # 添加一个头部标识
        final_comment = f"## 🤖 AI Code Review ({OPENROUTER_MODEL})\n\n{review_comment}"
        
        pr.create_issue_comment(final_comment)
        print("🎉 评论已发布到 GitHub。")

    except Exception as e:
        print(f"❌ 发生未捕获错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()