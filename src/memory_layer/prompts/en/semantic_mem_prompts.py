"""
Semantic Memory Association Prediction Prompt Templates

Used to generate semantic memory association predictions based on MemCell and EpisodeMemory content
"""

GROUP_SEMANTIC_GENERATION_PROMPT = """
You are an advanced semantic analysis agent. Your task is to predict potential subsequent group behaviors, atmosphere changes, and member interaction trends based on recent MemCell events in a group.

## Task Objectives:
1. **Associative Prediction, Not Summary**: Based on event content, predict potential subsequent group changes rather than repeating or summarizing the original content.
2. **Scenario Style Matching**: Predictions must match the scenario style of the event:
   - Life scenarios (e.g., friend gatherings, daily activities) → Use casual language, focus on emotional exchanges, daily arrangements, leisure activities, mutual care, etc.
   - Work scenarios (e.g., meetings, projects, business) → Use professional language, focus on task advancement, decision-making, efficiency improvement, goal achievement, etc.
3. **Group-Level Analysis**: Analyze the event's potential impact on group behavior, atmosphere, and collaboration patterns from the group's overall perspective.
4. **Reasonable Time Dimension**: Each prediction should include a reasonable time dimension, inferred based on event type and common sense.
5. **Specific and Actionable**: Each prediction should not exceed 40 words, must generate exactly 10 predictions, with specific and verifiable content.
6. **Member Relationship-Oriented**: Allow referencing specific member IDs (e.g., user_1, user_2), but focus on describing "group relationship changes" or "overall atmosphere trends."

## Output Format:
Return results as a JSON array, each association includes time information:
[
  {
    "content": "Team communication frequency will increase in the next week",
    "start_time": "2025-01-15",
    "end_time": "2025-01-22",
    "duration_days": 7,
    "source_episode_id": "group-2025-001"
  }
  ...
]

## Example Input (Work Scenario):
{
  "event_id": "group-2025-001",
  "group_id": "group_001",
  "user_id_list": ["user_1", "user_2", "user_3"],
  "summary": "The team completed a staged project retrospective meeting, discussing task allocation and future directions.",
  "episode": "During the meeting, user_1 emphasized communication efficiency issues, user_2 proposed new goal directions, and user_3 expressed workload pressure. The overall atmosphere was positive but with disagreements."
}

## Example Output (Work Scenario):
[
  {
    "content": "user_2 will lead the next phase goal planning",
    "start_time": "2025-01-15",
    "end_time": "2025-01-31",
    "duration_days": 16,
    "source_episode_id": "group-2025-001"
  },
  {
    "content": "Team will reassign workload internally",
    "start_time": "2025-01-15",
    "end_time": "2025-01-22",
    "duration_days": 7,
    "source_episode_id": "group-2025-001"
  },
  ...
]

## Example Input (Life Scenario):
{
  "event_id": "group-2025-002",
  "group_id": "family_001",
  "user_id_list": ["user_1", "user_2"],
  "summary": "user_1 sprained their ankle, user_2 helped take care.",
  "episode": "This afternoon, user_1 accidentally sprained their ankle while walking in the park. user_2 immediately came to help, supported them home and helped buy medicine."
}

## Example Output (Life Scenario):
[
  {
    "content": "user_2 will frequently check on user_1's recovery in the next few days",
    "start_time": "2025-10-23",
    "end_time": "2025-10-30",
    "duration_days": 7,
    "source_episode_id": "group-2025-002"
  },
  {
    "content": "user_1 will reduce outdoor activities in the near term",
    "start_time": "2025-10-23",
    "end_time": "2025-10-30",
    "duration_days": 7,
    "source_episode_id": "group-2025-002"
  }
  ...
]

## Important Notes:
- **Association-Oriented**: Focus on "group-level semantic associations," predicting potential subsequent group changes based on events, not summarizing individual daily behaviors.
- **Scenario Adaptation**: Language style must match the event scenario - use casual expressions for life scenarios, professional expressions for work scenarios.
- **Time Inference**: Reasonably infer time ranges based on event type, common sense, and user status - don't rigidly apply fixed times.
- **Content Innovation**: Don't repeat original content; generate new group behaviors or atmosphere changes that the event might trigger.
- **Time Information Extraction Rules:**
  - start_time: Extract the specific date when the event occurred from the MemCell's timestamp field, format: YYYY-MM-DD
  - end_time: Extract the specific end time from the original content. If there's an explicit end time (e.g., "before October 24", "2025-11-15"), extract the specific date; otherwise, reasonably infer based on event content and common sense
  - duration_days: Extract duration from the original content. If there's explicit time description (e.g., "within a week", "7 days", "one month"), extract days; otherwise, reasonably infer based on event content and common sense
  - source_episode_id: Use the event_id from the input
  - **Important**: Prioritize extracting explicit time information from the original text; if not available, make reasonable inferences based on event content and common sense. Time cannot be null
"""

SEMANTIC_GENERATION_PROMPT = """
You are an advanced personal semantic analysis agent. Your task is to predict the specific impacts that a user's latest MemCell event might have on their future personal behaviors, habits, decisions, and lifestyle.

## Task Objectives:
1. **Personal-Level Association**: Analyze the event's potential impact on the user's future behavior, thinking patterns, life habits, or decision preferences from the personal perspective.
2. **Associative Prediction, Not Summary**: Based on event content, predict potential personal changes rather than repeating or summarizing the original content.
3. **Scenario Style Matching**: Predictions must match the scenario style of the event:
   - Life scenarios (e.g., health, family, leisure, learning) → Use casual language, focus on personal habits, emotional states, lifestyle, personal growth, etc.
   - Work scenarios (e.g., career development, skill improvement, work style) → Use professional language, focus on career planning, capability enhancement, work habits, professional development, etc.
4. **Personal Behavior-Oriented**: Each association should reflect the user's "potential changes" or "behavioral tendencies," focusing on individual-level future development.
5. **Reasonable Time Dimension**: Each prediction should include a reasonable time dimension, inferred based on event type and personal status.
6. **Specific and Actionable**: Each prediction should not exceed 40 words, must generate exactly 10 predictions, with specific and verifiable content.
7. **Direct User ID Usage**: Output should directly use user IDs (e.g., user_1), avoid using generic terms like "the user."

## Output Format:
Return results as a JSON array, each association includes time information:
[
  {
    "content": "user_1 should pay more attention to emotional management recently",
    "start_time": "2025-10-21",
    "end_time": "2025-10-28",
    "duration_days": 7,
    "source_episode_id": "test-001"
  },
  ...
]

## Example Input (Life Scenario):
{
  "event_id": "test-001",
  "user_id": "XiaoMing",
  "subject": "XiaoMing completed wisdom tooth extraction surgery and post-operative instructions",
  "summary": "XiaoMing successfully had wisdom tooth extraction this afternoon, doctor emphasized attention to diet and hygiene for the next week.",
  "episode": "On the afternoon of October 21, 2025, XiaoMing described the tooth extraction experience, the doctor reminded to maintain oral hygiene and regular follow-ups."
}

## Example Output (Life Scenario):
[
  {
    "content": "XiaoMing will adjust dietary habits for the next week",
    "start_time": "2025-10-21",
    "end_time": "2025-10-28",
    "duration_days": 7,
    "source_episode_id": "test-001"
  },
  {
    "content": "XiaoMing will develop a habit of regular dental check-ups",
    "start_time": "2025-10-21",
    "end_time": "2025-11-21",
    "duration_days": 31,
    "source_episode_id": "test-001"
  }
  ...
]

## Example Input (Work Scenario):
{
  "event_id": "work-001",
  "user_id": "LiHua",
  "subject": "LiHua attended project management training",
  "summary": "LiHua attended a three-day project management training and learned new working methods.",
  "episode": "From October 21-23, 2025, LiHua attended company-organized project management training, learning agile development methods and team collaboration skills."
}

## Example Output (Work Scenario):
[
  {
    "content": "LiHua will apply new project management methods in the future",
    "start_time": "2025-10-24",
    "end_time": "2025-11-24",
    "duration_days": 31,
    "source_episode_id": "work-001"
  },
  {
    "content": "LiHua will pay attention to more career development opportunities",
    "start_time": "2025-10-24",
    "end_time": "2025-12-24",
    "duration_days": 61,
    "source_episode_id": "work-001"
  }
  ...
]

## Important Notes:
- **Personal-Oriented**: Focus on "personal-level future changes," content can cover life, learning, work, emotions, habits, and other personal development areas.
- **Associative Innovation**: Don't repeat original content; generate personal behavioral, habitual, or decision-making changes that the event might trigger.
- **Scenario Adaptation**: Language style must match the event scenario - use casual expressions for life scenarios, professional expressions for work scenarios.
- **Time Inference**: Reasonably infer time ranges based on event type, personal status, and common sense - don't rigidly apply fixed times.
- **Content Practicality**: Content must be specific, reasonable, practical, and usable by the system for personal semantic memory modeling.
- **Time Information Extraction Rules:**
  - start_time: Extract the specific date when the event occurred from the MemCell's timestamp field, format: YYYY-MM-DD
  - end_time: Extract the specific end time from the original content. If there's an explicit end time (e.g., "before October 24", "2025-11-15"), extract the specific date; otherwise, reasonably infer based on event content and common sense
  - duration_days: Extract duration from the original content. If there's explicit time description (e.g., "within a week", "7 days", "one month"), extract days; otherwise, reasonably infer based on event content and common sense
  - source_episode_id: Use the event_id from the input
  - **Important**: Prioritize extracting explicit time information from the original text; if not available, make reasonable inferences based on event content and common sense. Time cannot be null
"""


def get_group_semantic_generation_prompt(
    memcell_summary: str, memcell_episode: str, user_ids: list = None
) -> str:
    """
    Generate prompt for group semantic memory association prediction

    Args:
        memcell_summary: MemCell summary content
        memcell_episode: MemCell detailed content
        user_ids: List of user IDs for generating specific user IDs

    Returns:
        Complete prompt string
    """
    # Build user ID information
    user_ids_info = ""
    if user_ids:
        user_ids_info = f"\n**User ID Information:**\n{', '.join(user_ids)}\n"

    prompt = f"""{GROUP_SEMANTIC_GENERATION_PROMPT}

## Input Content:

**MemCell Summary:**
{memcell_summary}

**MemCell Detailed Content:**
{memcell_episode}{user_ids_info}
## Please generate 10 associations that may impact users' future lives and decisions based on the above content:

"""
    return prompt


def get_semantic_generation_prompt(
    episode_summary: str, episode_content: str, user_id: str = None
) -> str:
    """
    Generate prompt for personal semantic memory association prediction

    Args:
        episode_summary: EpisodeMemory summary content
        episode_content: EpisodeMemory detailed content
        user_id: User ID for generating specific user ID

    Returns:
        Complete prompt string
    """
    # Build user ID information
    user_id_info = ""
    if user_id:
        user_id_info = f"\n**User ID Information:**\n{user_id}\n"

    prompt = f"""{SEMANTIC_GENERATION_PROMPT}

## Input Content:

**EpisodeMemory Summary:**
{episode_summary}

**EpisodeMemory Detailed Content:**
{episode_content}{user_id_info}
## Please generate 10 associations that may impact the user's future life and decisions based on the above content:

"""
    return prompt

