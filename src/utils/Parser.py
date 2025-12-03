import re
from typing import List
from src.Config import Settings

class Parser:
    def __init__(self, settings: Settings):
        self.namingConvention = settings.imageNamingConvention
        self.regex = self._createRegexFromConvention(self.namingConvention, settings.imageExtension)

    def _createRegexFromConvention(self, convention: str, extensions: List[str]) -> re.Pattern:
        escaped = re.escape(convention)
        pattern = re.sub(r"\\{(\w+)\\}", r"(?P<\1>.+)", escaped)

        ext_pattern = "|".join(ext.lstrip(".") for ext in extensions)
        pattern = f"^{pattern}\\.({ext_pattern})$"

        print(f"Generated regex pattern: {pattern}")
        return re.compile(pattern)
        

    def parseFilename(self, filename: str) -> dict:
        match = re.match(self.regex, filename)
        if not match:
            raise ValueError(f"Filename '{filename}' does not match the naming convention.")
        return match.groupdict()