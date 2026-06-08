"""Google Gemini LLM client for CodeQ-Mate answer generation.

Provides an async-compatible client that wraps the Google Generative AI SDK
to generate source-grounded answers from retrieved code context.
"""

import os

import google.generativeai as genai


# Default model to use
DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiClient:
    """Async-compatible client for Google Gemini LLM.

    Wraps the google-generativeai SDK to provide a simple `generate(prompt)`
    interface compatible with the AnswerGenerator's expected LLM client protocol.

    Usage:
        client = GeminiClient(api_key="your-key")
        answer = await client.generate("Your prompt here")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_output_tokens: int = 2048,
    ) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY
                environment variable.
            model_name: Gemini model to use (default: gemini-1.5-flash).
            temperature: Sampling temperature (0.0-1.0, lower = more focused).
            max_output_tokens: Maximum tokens in the generated response.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Provide it via the api_key parameter "
                "or set the GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

    async def generate(self, prompt: str) -> str:
        """Generate a response from Gemini given a prompt.

        This method is async-compatible to work with the AnswerGenerator's
        retry logic. The underlying Gemini SDK call is synchronous but
        wrapped for compatibility.

        Args:
            prompt: The full prompt text to send to Gemini.

        Returns:
            The generated text response from Gemini.

        Raises:
            RuntimeError: If Gemini returns an empty response or errors out.
        """
        try:
            response = self.model.generate_content(prompt)

            if not response or not response.text:
                raise RuntimeError("Gemini returned an empty response")

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e
