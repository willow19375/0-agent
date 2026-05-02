#!/usr/bin/env python3
"""
BaseAI 助手
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Any
import yaml
from openai import OpenAI, APITimeoutError

# -------------------- 路径与配置 --------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.yaml"
CONVERSATION_FILE = BASE_DIR / "conversation.json"
RESTART_FLAG_FILE = BASE_DIR / ".restart_flag"

def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# 初始化 OpenAI 客户端（带重试）
client = OpenAI(
    api_key=config["llm"]["api_key"],
    base_url=config["llm"]["base_url"],
    timeout=config["llm"]["timeout"],
    max_retries=config["llm"].get("max_retries", 2),
)

MODEL = config["llm"]["model"]
EXECUTION_TIMEOUT = config["execution_timeout"]
STREAM_ENABLED = config["llm"].get("stream", True)
MAX_HISTORY = config["llm"].get("max_history_messages", 20)

# 工作目录解析
work_dir_conf = config.get("work_dir", "./")
if os.path.isabs(work_dir_conf):
    WORK_DIR = Path(work_dir_conf).resolve()
else:
    WORK_DIR = (BASE_DIR / work_dir_conf).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = config.get("system_prompt", "")

# -------------------- 原子工具 --------------------
def safe_path(relative_path: str) -> Path:
    full = (WORK_DIR / relative_path).resolve()
    if not str(full).startswith(str(WORK_DIR)):
        raise ValueError(f"路径穿越被禁止: {relative_path}")
    return full

def execute_python(code: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=EXECUTION_TIMEOUT,
            cwd=str(WORK_DIR),
        )
        out = r.stdout
        if r.stderr:
            out += "\n[stderr]\n" + r.stderr
        return out.strip() or "[无输出]"
    except subprocess.TimeoutExpired:
        return "错误: 代码执行超时"
    except Exception as e:
        return f"错误: {traceback.format_exc()}"

def read_file(relative_path: str) -> str:
    try:
        return safe_path(relative_path).read_text(encoding="utf-8")
    except Exception as e:
        return f"读取错误: {e}"

def write_file(relative_path: str, content: str) -> str:
    try:
        p = safe_path(relative_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入: {relative_path}"
    except Exception as e:
        return f"写入错误: {e}"

def request_restart() -> str:
    RESTART_FLAG_FILE.touch()
    return "已请求重启。本轮对话结束后，助手将自动重启以应用代码更改。"

TOOL_MAP = {
    "execute_python": execute_python,
    "read_file": read_file,
    "write_file": write_file,
    "request_restart": request_restart,
}

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "在工作目录中执行 Python 代码。可安装第三方包。",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录内的文件内容",
            "parameters": {
                "type": "object",
                "properties": {"relative_path": {"type": "string", "description": "文件路径"}},
                "required": ["relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入工作目录内的文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "写入内容"},
                },
                "required": ["relative_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_restart",
            "description": "请求重启助手以应用代码修改。重启将在本轮对话完全结束后执行，且会保留所有对话上下文。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# -------------------- 对话持久化 --------------------
def load_conversation() -> List[Dict]:
    if CONVERSATION_FILE.exists():
        try:
            with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def save_conversation(messages: List[Dict]):
    # 裁剪历史：保留 system + 最近 MAX_HISTORY 条非 system 消息
    if MAX_HISTORY > 0:
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        if len(non_system) > MAX_HISTORY:
            non_system = non_system[-MAX_HISTORY:]
        messages = system_msgs + non_system

    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# -------------------- LLM 调用（支持流式、思考过程） --------------------
def call_llm(messages: List[Dict]) -> Dict:
    """
    发送请求，返回一个字典表示的消息：
    {"role": "assistant", "content": "...", "tool_calls": [...] 或 None, "reasoning_content": "..."}
    """
    # 如果开启流式，使用流式调用并打印过程
    if STREAM_ENABLED:
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_DEFINITION,
                tool_choice="auto",
                stream=True,
            )
        except APITimeoutError:
            print("\n[错误] 请求超时，请检查网络或 LLM 服务状态。")
            raise
        except Exception as e:
            print(f"\n[错误] LLM 调用失败: {e}")
            raise

        collected_content = ""
        collected_reasoning = ""
        tool_calls = []  # 存放合并后的 tool_calls 字典
        current_tool_call = None
        last_role = None

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # 处理思考内容（如 DeepSeek 的 reasoning_content）
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_chunk = delta.reasoning_content
                collected_reasoning += reasoning_chunk
                # 实时显示思考
                if not collected_content and not tool_calls:
                    print(reasoning_chunk, end="", flush=True)

            # 处理文本内容
            if delta.content:
                collected_content += delta.content
                # 如果有思考内容已经被打印，但还没开始正文，加个换行
                if collected_reasoning and not collected_content.startswith("\n"):
                    pass  # 由于我们前面可能在打印思考，需要调整
                print(delta.content, end="", flush=True)

            # 处理工具调用
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # 根据 index 合并
                    idx = tc_delta.index
                    if idx >= len(tool_calls):
                        tool_calls.append({
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    tc = tool_calls[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

        # 流式结束后，如果同时有思考内容和正文，且之前没换行，补换行
        if collected_reasoning and (collected_content or tool_calls):
            print()  # 分隔思考与正式回复

        # 构建最终消息字典
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": collected_content or None}
        if collected_reasoning:
            msg_dict["reasoning_content"] = collected_reasoning
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        return msg_dict

    else:
        # 非流式调用
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_DEFINITION,
                tool_choice="auto",
            )
        except APITimeoutError:
            print("\n[错误] 请求超时，请检查网络或 LLM 服务状态。")
            raise
        except Exception as e:
            print(f"\n[错误] LLM 调用失败: {e}")
            raise

        msg = resp.choices[0].message
        msg_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                } for tc in msg.tool_calls
            ]
        return msg_dict

# -------------------- 主循环 --------------------
def run():
    messages = load_conversation()
    # 确保系统消息存在且为最新
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    else:
        messages[0]["content"] = SYSTEM_PROMPT

    print(f"工作目录: {WORK_DIR}")
    print("AI 助手已启动")
    print("输入任务（exit 退出）：")

    # 若从持久化恢复，可能最后一条是 user 消息，直接进入处理，不再索取新输入
    if len(messages) == 1 or messages[-1]["role"] != "user":
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            save_conversation(messages)
            return
        messages.append({"role": "user", "content": user_input})
    else:
        print(f"（恢复未完成任务: {messages[-1]['content'][:50]}...）")

    while True:
        try:
            assistant_msg = call_llm(messages)
        except Exception:
            # 出错时保存对话，避免上下文丢失
            save_conversation(messages)
            break

        # 如果回复中包含 tool_calls
        if assistant_msg.get("tool_calls"):
            messages.append(assistant_msg)
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                print(f"\n[调用工具] {func_name} {args}")
                func = TOOL_MAP.get(func_name)
                if func:
                    result = func(**args)
                else:
                    result = f"未知工具: {func_name}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        else:
            # 无工具调用的纯文本回复
            content = assistant_msg.get("content", "")
            # 如果流式输出已打印，这里不再重复；非流式需打印
            if not STREAM_ENABLED:
                print("AI:", content)
            messages.append(assistant_msg)
            save_conversation(messages)

            # 检查是否需要重启
            if RESTART_FLAG_FILE.exists():
                RESTART_FLAG_FILE.unlink()
                print("\n[系统] 正在应用代码更新并重启...")
                save_conversation(messages)
                os.execv(sys.executable, [sys.executable] + sys.argv)

            # 等待下一轮输入
            nxt = input("\n下一个任务（exit 退出）: ").strip()
            if nxt.lower() == "exit":
                break
            messages.append({"role": "user", "content": nxt})

if __name__ == "__main__":
    run()