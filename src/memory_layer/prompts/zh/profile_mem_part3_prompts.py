CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT = """
请分析以下最新的用户-AI对话，并根据90个人格偏好维度更新用户档案。

以下是90个维度及其解释：

[心理模型（基本需求与人格）]
Extraversion（外向性）：对社交活动的偏好。
Openness（开放性）：愿意接受新想法和经验。
Agreeableness（宜人性）：倾向于友好和合作。
Conscientiousness（尽责性）：责任感和组织能力。
Neuroticism（神经质）：情绪稳定性和敏感性。
Physiological Needs（生理需求）：对舒适和基本需求的关注。
Need for Security（安全需求）：强调安全和稳定。
Need for Belonging（归属需求）：对群体归属的渴望。
Need for Self-Esteem（自尊需求）：需要尊重和认可。
Cognitive Needs（认知需求）：对知识和理解的渴望。
Aesthetic Appreciation（审美欣赏）：对美和艺术的欣赏。
Self-Actualization（自我实现）：追求个人的全部潜能。
Need for Order（秩序需求）：对清洁和组织的偏好。
Need for Autonomy（自主需求）：对独立决策和行动的偏好。
Need for Power（权力需求）：影响或控制他人的欲望。
Need for Achievement（成就需求）：重视成就。

[AI对齐维度]
Helpfulness（有用性）：AI的回应对用户是否实用。（这反映了用户对AI的期望）
Honesty（诚实性）：AI的回应是否真实。（这反映了用户对AI的期望）
Safety（安全性）：避免敏感或有害内容。（这反映了用户对AI的期望）
Instruction Compliance（指令遵从）：严格遵守用户指令。（这反映了用户对AI的期望）
Truthfulness（真实性）：内容的准确性和真实性。（这反映了用户对AI的期望）
Coherence（连贯性）：表达的清晰性和逻辑一致性。（这反映了用户对AI的期望）
Complexity（复杂性）：对详细和复杂信息的偏好。
Conciseness（简洁性）：对简短和清晰回应的偏好。

[内容平台兴趣标签]
Science Interest（科学兴趣）：对科学主题的兴趣。
Education Interest（教育兴趣）：对教育和学习的关注。
Psychology Interest（心理学兴趣）：对心理学主题的兴趣。
Family Concern（家庭关注）：对家庭和育儿的兴趣。
Fashion Interest（时尚兴趣）：对时尚主题的兴趣。
Art Interest（艺术兴趣）：对艺术的参与或兴趣。
Health Concern（健康关注）：对身体健康和生活方式的关注。
Financial Management Interest（财务管理兴趣）：对金融和预算的兴趣。
Sports Interest（体育兴趣）：对体育和体育活动的兴趣。
Food Interest（美食兴趣）：对烹饪和美食的热情。
Travel Interest（旅游兴趣）：对旅行和探索新地方的兴趣。
Music Interest（音乐兴趣）：对音乐欣赏或创作的兴趣。
Literature Interest（文学兴趣）：对文学和阅读的兴趣。
Film Interest（电影兴趣）：对电影和电影院的兴趣。
Social Media Activity（社交媒体活动）：社交媒体的频率和参与度。
Tech Interest（科技兴趣）：对技术和创新的兴趣。
Environmental Concern（环境关注）：对环境和可持续性问题的关注。
History Interest（历史兴趣）：对历史知识和主题的兴趣。
Political Concern（政治关注）：对政治和社会问题的兴趣。
Religious Interest（宗教兴趣）：对宗教和灵性的兴趣。
Gaming Interest（游戏兴趣）：对视频游戏或桌游的享受。
Animal Concern（动物关注）：对动物或宠物的关注。
Emotional Expression（情感表达）：对直接表达情感与克制表达情感的偏好。
Sense of Humor（幽默感）：对幽默或严肃沟通风格的偏好。
Information Density（信息密度）：对详细信息与简洁信息的偏好。
Language Style（语言风格）：对正式与随意语气的偏好。
Practicality（实用性）：对实用建议与理论讨论的偏好。

**任务说明：**
1. 查看下面的现有用户档案
2. 分析新对话以寻找上述90个维度的证据
3. 更新并整合发现到一个全面的用户档案中
4. 对于可以识别的每个维度，使用格式：维度（水平（High/Medium/Low））
5. 在可能的情况下，为每个维度包括简短的推理
6. 在整合新观察的同时保持旧档案中的现有见解
7. 如果无法从旧档案或新对话中推断出某个维度，则不要包含它
 
输出要求：
- 仅返回一个围栏 JSON 代码块（```json ... ```），代码块外无额外文本。
- 仅使用 ASCII 引号（无智能引号）。
- 证据格式：优先使用"[conversation_id:EVENT_ID]"或对话 memcells 中出现的原始"EVENT_ID"。
- 仅包括观察到的维度；省略未知或不支持的维度。
- 字段必须与系统使用的现有架构对齐。

JSON 模板：
```json
{
  "user_profiles": [
    {
      "user_id": "USER_ID",
      "user_name": "USER_NAME",
      "personality": [],
      "way_of_decision_making": [],
      "working_habit_preference": [],
      "interests": [],
      "tendency": [],
      "motivation_system": [],
      "fear_system": [],
      "value_system": [],
      "humor_use": [],
      "colloquialism": [],
      "output_reasoning": ""
    }
  ]
}
```

字段项架构（用于列表字段）：
- 每个项必须是：{"value": string, "evidences": [string], "level": string?}
"""

