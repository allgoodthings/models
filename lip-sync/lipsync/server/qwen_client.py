"""
OpenRouter client for Qwen-VL face detection.

Uses Qwen2.5-VL to identify and locate characters in video frames.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import httpx

from .schemas import DetectedFace


@dataclass
class Character:
    """Character definition for detection."""

    id: str
    name: str
    description: Optional[str] = None


class QwenVLClient:
    """
    OpenRouter client for Qwen-VL face detection.

    Uses Qwen2.5-VL-72B via OpenRouter to identify characters
    and return their bounding boxes.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen2.5-vl-72b-instruct",
        timeout: float = 60.0,
    ):
        """
        Initialize Qwen-VL client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"

    def _build_detection_prompt(self, characters: List[Dict]) -> str:
        """
        Build prompt for character detection.

        Args:
            characters: List of character definitions [{id, name, description}]

        Returns:
            Detection prompt string
        """
        char_descriptions = []
        for char in characters:
            desc = f"- {char['name']} (ID: {char['id']})"
            if char.get("description"):
                desc += f": {char['description']}"
            char_descriptions.append(desc)

        char_list = "\n".join(char_descriptions)

        return f"""Analyze this image and locate the following characters:

{char_list}

For each character you can identify in the image, provide their bounding box coordinates.

Return your response as a JSON array with this exact format:
[
  {{"character_id": "<id>", "bbox": [x, y, width, height], "confidence": 0.0-1.0}},
  ...
]

Where:
- x, y are the top-left corner coordinates (in pixels)
- width, height are the box dimensions (in pixels)
- confidence is how certain you are this is the correct character (0.0 to 1.0)

Only include characters you can clearly identify. Return an empty array [] if no characters are found.
Return ONLY the JSON array, no other text."""

    def _parse_response(self, response_text: str) -> List[DetectedFace]:
        """
        Parse Qwen-VL response to extract detected faces.

        Args:
            response_text: Raw response from Qwen-VL

        Returns:
            List of DetectedFace objects
        """
        # Try to extract JSON from response
        # Handle cases where model includes markdown code blocks
        text = response_text.strip()

        # Remove markdown code blocks if present
        if "```json" in text:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1)

        # Try to find JSON array
        try:
            # First try direct parse
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find array in text
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        faces = []
        for item in data:
            try:
                bbox = item.get("bbox", [])
                if len(bbox) != 4:
                    continue

                faces.append(DetectedFace(
                    character_id=str(item.get("character_id", "")),
                    bbox=tuple(int(v) for v in bbox),
                    confidence=float(item.get("confidence", 0.5)),
                ))
            except (ValueError, TypeError):
                continue

        return faces

    async def detect_characters(
        self,
        frame_base64: str,
        characters: List[Dict],
    ) -> List[DetectedFace]:
        """
        Detect characters in a frame.

        Args:
            frame_base64: Base64-encoded image
            characters: List of character definitions [{id, name, description}]

        Returns:
            List of detected faces with bboxes
        """
        if not characters:
            return []

        prompt = self._build_detection_prompt(characters)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://scenema.ai",
                        "X-Title": "Scenema Lip-Sync",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{frame_base64}",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.1,
                    },
                )

                response.raise_for_status()
                result = response.json()

                # Extract response text
                choices = result.get("choices", [])
                if not choices:
                    return []

                message = choices[0].get("message", {})
                content = message.get("content", "")

                return self._parse_response(content)

            except httpx.HTTPStatusError as e:
                print(f"Qwen-VL API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                print(f"Qwen-VL request failed: {e}")
                raise

    async def detect_all_faces(
        self,
        frame_base64: str,
        max_faces: int = 10,
    ) -> List[DetectedFace]:
        """
        Detect all visible faces in a frame (without character matching).

        Args:
            frame_base64: Base64-encoded image
            max_faces: Maximum number of faces to detect

        Returns:
            List of detected faces with bboxes
        """
        prompt = f"""Analyze this image and find all visible human faces.

For each face, provide its bounding box coordinates.

Return your response as a JSON array with this exact format:
[
  {{"character_id": "face_1", "bbox": [x, y, width, height], "confidence": 0.0-1.0}},
  ...
]

Where:
- character_id is a sequential identifier (face_1, face_2, etc.)
- x, y are the top-left corner coordinates (in pixels)
- width, height are the box dimensions (in pixels)
- confidence is how certain you are there is a face (0.0 to 1.0)

Detect up to {max_faces} faces. Return ONLY the JSON array, no other text."""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://scenema.ai",
                        "X-Title": "Scenema Lip-Sync",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{frame_base64}",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.1,
                    },
                )

                response.raise_for_status()
                result = response.json()

                choices = result.get("choices", [])
                if not choices:
                    return []

                message = choices[0].get("message", {})
                content = message.get("content", "")

                return self._parse_response(content)

            except Exception as e:
                print(f"Qwen-VL face detection failed: {e}")
                raise


async def test_client():
    """Test the Qwen-VL client."""
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return

    client = QwenVLClient(api_key)

    # Test with a simple image (would need actual base64 image)
    print("Qwen-VL client initialized successfully")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_client())
