"""根据 Agent 配置生成能力卡片的通用工具函数.

该模块不直接依赖具体的 FastAPI 服务, 仅依赖:
- Agent 配置字典
- 一个兼容 OpenAI Async 客户端的 chat.completions 接口
- 模型名称字符串

用于 server 和 desktop 两端的 service 调用.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from .logger import logger


class AgentAbilitiesGenerationError(Exception):
    """在生成 Agent 能力列表时出现的受控异常."""


def _normalize_id(raw: str) -> str:
    """将任意字符串规范化为 kebab-case id.

    - 转小写
    - 非字母数字统一替换为 '-'
    - 合并重复的 '-'
    - 去掉首尾 '-'
    """

    value = (raw or "").strip().lower()
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    value = value.strip("-")
    return value


def _format_name_list(label: str, values: Iterable[str] | None, max_count: int = 10) -> str:
    names = [str(v) for v in (values or []) if str(v).strip()]
    if not names:
        return f"{label}：无"
    display = "、".join(names[:max_count])
    if len(names) > max_count:
        display += " 等"
    return f"{label}：{display}"


def _build_context_summary(context: Optional[Dict[str, Any]]) -> str:
    if not context or not isinstance(context, dict):
        return "当前没有额外上下文。"

    parts: List[str] = []
    workspace = context.get("workspace") or context.get("workspace_name")
    if workspace:
        parts.append(f"- 当前工作空间：{workspace}")

    current_file = context.get("current_file") or context.get("file_path")
    if current_file:
        parts.append(f"- 当前文件：{current_file}")

    if not parts:
        return "当前没有额外上下文。"
    return "\n".join(parts)


async def generate_agent_abilities_from_config(
    agent_config: Dict[str, Any],
    context: Optional[Dict[str, Any]],
    client: Any,
    model: str,
    language: str = "zh",
) -> List[Dict[str, str]]:
    """基于 Agent 配置调用 LLM 生成能力卡片列表.

    Args:
        agent_config: Agent 配置字典, 来自现有 Agent 服务.
        context: 可选上下文(预留字段), 例如当前 workspace / 文件等.
        client: 兼容 OpenAI Async 客户端的实例, 需要支持
            `await client.chat.completions.create(...)` 接口.
        model: 使用的模型名称.
        language: 界面语言, 当前版本仍固定生成中文文案, 预留扩展.

    Returns:
        List[Dict[str, str]]: 每个元素包含 id/title/description/promptText.

    Raises:
        AgentAbilitiesGenerationError: 当模型调用或结果解析失败时抛出.
    """

    if not client:
        raise AgentAbilitiesGenerationError("模型客户端未配置")

    name = agent_config.get("name") or agent_config.get("id") or "Sage 助手"
    description = (
        agent_config.get("description")
        or agent_config.get("systemPrefix")
        or agent_config.get("system_prefix")
        or "这是一个通用智能助手。"
    )

    tools = agent_config.get("availableTools") or agent_config.get("available_tools") or []
    skills = agent_config.get("availableSkills") or agent_config.get("available_skills") or []
    workflows_cfg = agent_config.get("availableWorkflows") or agent_config.get("available_workflows") or {}

    if isinstance(workflows_cfg, dict):
        workflow_names: List[str] = list(workflows_cfg.keys())
    elif isinstance(workflows_cfg, list):
        workflow_names = [str(x) for x in workflows_cfg]
    else:
        workflow_names = []

    tools_line = _format_name_list("可用工具", tools)
    skills_line = _format_name_list("可用技能", skills)
    workflows_line = _format_name_list("可用工作流", workflow_names)

    context_summary = _build_context_summary(context)

    system_prompt = f"""
你是一个根据 Agent 配置生成「能力卡片」(ability cards) 的助手。

你的任务：
- 阅读我提供的 Agent 配置信息和可用工具/技能/工作流/上下文，
- 总结这个 Agent 最典型、最有代表性的 6-8 条“能帮用户完成什么事情”的能力。

输出要求（非常重要）：
- 必须只输出 JSON 字符串，不能包含任何额外说明文字。
- JSON 结构为：
  {{
    "items": [
      {{
        "id": "kebab-case-id",
        "title": "短标题（中文）",
        "description": "1-2 句中文说明",
        "promptText": "用户可以直接发送给助手的自然语言提问（中文）"
      }}
    ]
  }}

字段要求：
- id：全部小写，使用短横线分隔（kebab-case），例如 "code-review"、"optimize-sql-query"。
- title：概括该能力做什么，简短有力，使用中文。
- description：用 1-2 句话解释该能力的用途和适用场景，使用中文。
- promptText：写成用户可以直接复制粘贴发送的中文自然语言提问，可以适当预留「请在这里粘贴 xxx」之类的提示。

内容要求：
- 严格基于提供的 Agent 描述、可用工具/技能/工作流 和 上下文 进行总结。
- 不要编造明显不存在的专业能力或工具。
- 尽量覆盖不同类型/场景的能力，而不是重复的说法。
- 总共输出 6-8 条能力项，不要少于 6 条，也不要多于 8 条。

当前界面语言标记为 "{language}"，但本次请一律使用简体中文输出。

请牢记：只输出符合上述结构的 JSON，不能在 JSON 外多说任何一句话。
""".strip()

    user_prompt = f"""
下面是当前 Agent 的配置信息，请基于这些信息生成能力卡片：

- Agent 名称：{name}
- Agent 简介：{description}

{tools_line}
{skills_line}
{workflows_line}

可选上下文信息（如果有）：
{context_summary}

请根据以上信息，生成 6-8 条最典型、最有代表性的能力项，覆盖该 Agent 最适合解决的问题类型。
""".strip()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500,
        )
    except Exception as e:  # pragma: no cover - 具体异常类型由底层 SDK 决定
        logger.error(f"生成 Agent 能力列表时调用模型失败: {e}")
        raise AgentAbilitiesGenerationError("调用模型失败，请稍后重试") from e

    # 解析模型返回的 JSON
    try:
        choice = response.choices[0].message
    except Exception as e:  # pragma: no cover
        logger.error(f"解析模型返回结果失败: {e}")
        raise AgentAbilitiesGenerationError("模型返回结果格式不正确") from e

    data_obj: Any
    parsed = getattr(choice, "parsed", None)
    if parsed is not None:
        data_obj = parsed
    else:
        content = getattr(choice, "content", None)
        if isinstance(content, list):
            # 兼容部分 SDK 将 content 表示为片段列表的情况
            content_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            content_text = str(content or "")

        try:
            data_obj = json.loads(content_text)
        except Exception as e:  # pragma: no cover
            logger.error(
                "解析能力卡 JSON 失败: {} | 原始内容开头: {}".format(
                    e, content_text[:500]
                )
            )
            raise AgentAbilitiesGenerationError("解析模型返回的能力列表失败") from e

    if not isinstance(data_obj, dict) or "items" not in data_obj:
        raise AgentAbilitiesGenerationError("能力列表结果缺少 items 字段")

    items_raw = data_obj.get("items") or []
    if not isinstance(items_raw, list):
        raise AgentAbilitiesGenerationError("能力列表 items 字段格式不正确")

    results: List[Dict[str, str]] = []
    seen_ids: set[str] = set()

    for raw in items_raw:
        if not isinstance(raw, dict):
            continue

        raw_id = str(raw.get("id") or "").strip()
        title = str(raw.get("title") or "").strip()
        desc = str(raw.get("description") or "").strip()
        prompt = str(raw.get("promptText") or "").strip()

        if not (raw_id and title and desc and prompt):
            continue

        norm_id = _normalize_id(raw_id)
        if not norm_id:
            continue

        if norm_id in seen_ids:
            base = norm_id
            suffix = 2
            while f"{base}-{suffix}" in seen_ids:
                suffix += 1
            norm_id = f"{base}-{suffix}"

        seen_ids.add(norm_id)
        results.append(
            {
                "id": norm_id,
                "title": title,
                "description": desc,
                "promptText": prompt,
            }
        )

        if len(results) >= 8:
            break

    if not results:
        raise AgentAbilitiesGenerationError("未生成任何有效的能力项")

    if len(results) < 4:
        logger.warning(
            "生成的能力项少于预期数量: {} 条".format(len(results))
        )

    logger.info(
        "成功为 Agent 生成 {} 条能力项".format(len(results))
    )

    return results
