import re
import logging
from pydantic import ValidationError
from schemas.response import AgentResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_json(text: str) -> str:
    # Grab first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)

def enforce_agent_response(raw_text: str) -> AgentResponse:
    json_text = extract_json(raw_text)
    return AgentResponse.model_validate_json(json_text)

def build_repair_prompt(error: Exception, raw_text: str) -> str:
    return f"""
Your previous response was INVALID.

Error:
{error}

You MUST return ONLY valid JSON exactly matching:
{{
  "answer": string,
  "sources": string[]
}}

Rules:
- sources must contain ONLY URLs
- sources must be URLs from Tavily results only (do not invent links)
- no markdown, no extra keys, no text outside JSON

Here was your invalid output:
{raw_text}
""".strip()

def enforce_with_retries(agent, messages, max_retries: int = 3) -> AgentResponse:
    last_err = None
    last_raw = None

    logger.info(f"Starting structured output enforcement with max_retries={max_retries}")

    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}: Invoking LLM...")
        ai = agent.invoke(messages)
        last_raw = ai.content
        logger.debug(f"Raw LLM response (first 200 chars): {last_raw[:200]}...")

        try:
            logger.info(f"Attempt {attempt}: Parsing structured response...")
            result = enforce_agent_response(last_raw)
            logger.info(f"Attempt {attempt}: Successfully parsed structured response")
            return result
        except (ValidationError, ValueError) as e:
            last_err = e
            logger.warning(f"Attempt {attempt}: Failed to parse response - {type(e).__name__}: {str(e)}")
            if attempt < max_retries:
                logger.info(f"Attempt {attempt}: Retrying with repair prompt...")
                messages = messages + [{"role": "user", "content": build_repair_prompt(e, last_raw)}]
            else:
                logger.error(f"Attempt {attempt}: Max retries reached. Giving up.")

    raise RuntimeError(f"Failed structured output after {max_retries} retries. Last error: {last_err}\nRaw:\n{last_raw}")
