from config import MAX_HISTORY, DEFAULT_LANGUAGE


class Memory:

    def __init__(self):
        self.history = []
        self.language = DEFAULT_LANGUAGE
        self.input_script = "Unknown"

    def set_language(self, language: str):
        self.language = language

    def get_language(self):
        return self.language

    def set_input_script(self, script: str):
        self.input_script = script or "Unknown"

    def get_input_script(self):
        return self.input_script

    def add_user(self, text: str):
        self.history.append({
            "role": "user",
            "content": text
        })
        self.trim()

    def add_assistant(self, text: str):
        self.history.append({
            "role": "assistant",
            "content": text
        })
        self.trim()

    def messages(self):
        return self.history

    def clear(self):
        self.history = []
        self.language = DEFAULT_LANGUAGE
        self.input_script = "Unknown"

    def state(self):
        return {
            "conversation_language": self.language,
            "language": self.language,
            "input_script": self.input_script,
            "history": self.history,
        }

    def trim(self):
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-MAX_HISTORY * 2:]