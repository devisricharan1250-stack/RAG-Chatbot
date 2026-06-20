"""
A thin wrapper around the language model that actually writes the answers.

The rest of the app only calls `generate(system, prompt)` and doesn't care
which provider is behind it. To switch providers, change LLM_PROVIDER in
config.py / .env -- no other code needs to change.
"""
import config


def generate(system: str, prompt: str) -> str:
    """Send a system instruction + user prompt to the configured LLM and
    return the plain-text answer."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "anthropic":
        return _anthropic(system, prompt)
    if provider == "openai":
        return _openai(system, prompt)
    if provider == "ollama":
        return _ollama(system, prompt)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _anthropic(system: str, prompt: str) -> str:
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    # The response content is a list of blocks; we want the text from the first.
    return message.content[0].text


def _openai(system: str, prompt: str) -> str:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def _ollama(system: str, prompt: str) -> str:
    """Talk to a locally running Ollama server. Free, private, no API key."""
    import requests

    response = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
