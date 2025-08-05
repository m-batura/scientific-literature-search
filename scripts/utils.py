import re


def load_text_from_path(text_path):
    try:
        with open(text_path, "r", encoding="utf-8") as file:
            template = file.read()
        return template
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {text_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")


# Regular expression to find JSON blocks enclosed in ```json ... ```
_md_json = r"```json\s*([\s\S]*?)\s*```"

def extract_json_from_markdown(text):

    match = re.search(_md_json, text)

    if match:
        json_text = match.group(1).strip()
        return json_text
    else:
        return None

