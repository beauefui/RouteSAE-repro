import argparse
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from openai import AzureOpenAI

from .utils import save_json

logger = logging.getLogger(__name__)


class Interpreter:
    """GPT-4o based interpretation of SAE features."""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        logger.info("Initialized Interpreter")

    def calculate_cost(self, input_text: str, output_text: str) -> float:
        """Calculate API cost based on token counts."""
        encoding = tiktoken.encoding_for_model(self.cfg.engine)
        num_input_tokens = len(encoding.encode(input_text))
        num_output_tokens = len(encoding.encode(output_text))
        if self.cfg.engine == 'gpt-4o':
            return num_input_tokens * 2.5 / 1_000_000 + num_output_tokens * 10 / 1_000_000
        elif self.cfg.engine == 'gpt-4o-mini':
            return num_input_tokens * 0.15 / 1_000_000 + num_output_tokens * 0.6 / 1_000_000
        else:
            return 0.0
    
    def construct_prompt(self, tokens_info: List[Dict[str, Any]]) -> str:
        """Construct GPT-4o prompt for feature interpretation."""
        prompt = (
            'We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text.\n'
            'Each token\'s activation value indicates its relevance to the feature, with higher values showing stronger association. Features are categorized as:\\n'
            'A. Low-level features, which are associated with word-level polysemy disambiguation (e.g., "crushed things", "Europe").\n'
            'B. High-level features, which are associated with long-range pattern formation (e.g., "enumeration", "one of the [number/quantifier]")\n'
            'C. Undiscernible features, which are associated with noise or irrelevant patterns.\n\n'
            'Your task is to classify the feature as low-level, high-level or undiscernible and give this feature a monosemanticity score based on the following scoring rubric:\n'
            'Activation Consistency\n'
            '5: Clear pattern with no deviating examples\n'
            '4: Clear pattern with one or two deviating examples\n'
            '3: Clear overall pattern but quite a few examples not fitting that pattern\n'
            '2: Broad consistent theme but lacking structure\n'
            '1: No discernible pattern\n'
            'Consider the following activations for a feature in the neural network.\n\n'
        )
        for info in tokens_info:
            prompt += f"Token: {info['token']} | Activation: {info['activation']} | Context: {info['context']}\n\n"
        prompt += (
            'Provide your response in the following fixed format:\n'
            'Feature category: [Low-level/High-level/Undiscernible]\n'
            'Score: [5/4/3/2/1]\n'
            'Explanation: [Your brief explanation]\n'
        )
        return prompt

    def chat_completion(self, client: AzureOpenAI, prompt: str, max_retry: int = 3) -> str:
        """Call GPT-4o API with retry logic."""
        if client is None:
            raise ValueError('OpenAI client is not initialized')
        
        for attempt in range(1, max_retry + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are an assistant that helps explain the latent semantics of language models.',
                        },
                        {'role': 'user', 'content': prompt},
                    ],
                    model=self.cfg.engine,
                    max_tokens=128,  
                    temperature=0.1,
                )
                response_content = chat_completion.choices[0].message.content
                if response_content is None:
                    raise ValueError('API returned None response')
                return response_content.strip()
            except Exception as e:
                logger.warning(f"API call attempt {attempt}/{max_retry} failed: {e}")
                if attempt == max_retry:
                    logger.error('Failed to get response from OpenAI API after all retries')
                    raise
        raise RuntimeError('Failed to get response from OpenAI API')
    
    def run(
        self, 
        data_path: Optional[str] = None, 
        sample_latents: int = 100, 
        output_path: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Run GPT-4o interpretation on sampled features.
        
        Returns:
            (avg_score, low_level_score, high_level_score)
        """
        if data_path is None:
            data_path = self.cfg.data_path

        if output_path is None:
            output_path = os.path.join(
                self.cfg.output_dir, 
                f'interp_{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}.json'
            )

        logger.info(f"Loading context data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        latent_context_map = data.get('latent_context_map', {})
        all_latents = list(latent_context_map.keys())
        sample_size = min(sample_latents, len(all_latents))
        sampled_indices = random.sample(range(len(all_latents)), sample_size)
        sampled_latents = [all_latents[i] for i in sorted(sampled_indices)]

        logger.info(f"Sampled {sample_size} features for interpretation")
        logger.info(f"Initializing OpenAI client (engine: {self.cfg.engine})")
        
        client = AzureOpenAI(
            azure_endpoint=self.cfg.api_base,
            api_version=self.cfg.api_version,
            api_key=self.cfg.api_key,
        )

        cost = 0.0
        results = {}
        total_score = 0.0
        scored_features = 0

        low_level_features = 0
        low_level_total_score = 0.0
        high_level_features = 0
        high_level_total_score = 0.0

        pattern = re.compile(
            r"Feature category:\s*(?P<category>low-level|high-level|undiscernible)\s*\n"
            r"Score:\s*(?P<score>[1-5])\s*\n"
            r"Explanation:\s*(?P<explanation>.+)",
            re.IGNORECASE | re.DOTALL,
        )

        for idx, latent in enumerate(sampled_latents, 1):
            try:
                latent_id = int(latent)
            except ValueError:
                logger.warning(f"Invalid latent ID {latent}. Skipping.")
                continue
            
            token_contexts = latent_context_map[latent]
            tokens_info = []
            for token_class, contexts in token_contexts.items():
                for context in contexts:
                    token = token_class
                    if token.startswith('ġ'):
                        token = ' ' + token[1:]
                    tokens_info.append({
                        'token': token,
                        'context': context['context'],
                        'activation': context['activation'],
                    })

            prompt = self.construct_prompt(tokens_info)
            try:
                response = self.chat_completion(client, prompt)
                cost += self.calculate_cost(prompt, response)

                match = pattern.search(response)
                if match:
                    category = match.group('category').strip().lower()
                    score = int(match.group('score'))
                    explanation = match.group('explanation').strip()

                    if 1 <= score <= 5 and category in ['low-level', 'high-level', 'undiscernible']:
                        results[latent_id] = {
                            'category': category,
                            'score': score,
                            'explanation': explanation,
                        }
                        total_score += score
                        scored_features += 1

                        if category == 'low-level':
                            low_level_features += 1
                            low_level_total_score += score
                        elif category == 'high-level':
                            high_level_features += 1
                            high_level_total_score += score

            except Exception as e:
                logger.error(f"Error processing latent {latent_id}: {e}")
                continue
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{sample_size} features")

        avg_score = total_score / scored_features if scored_features > 0 else 0.0
        low_level_score = low_level_total_score / low_level_features if low_level_features > 0 else 0.0
        high_level_score = high_level_total_score / high_level_features if high_level_features > 0 else 0.0

        logger.info(f"Interpretation complete: {scored_features} features scored")
        logger.info(f"Avg score: {avg_score:.2f}, Low-level: {low_level_score:.2f}, High-level: {high_level_score:.2f}")
        logger.info(f"Total API cost: ${cost:.4f}")
        
        output_data = {
            'cost': cost,
            'engine': self.cfg.engine,
            'features_scored': scored_features,
            'average_score': avg_score,
            'low_level_features': low_level_features,
            'low_level_score': low_level_score,
            'high_level_features': high_level_features,
            'high_level_score': high_level_score,
            'results': results,
        }
        save_json(output_data, output_path)
        return avg_score, low_level_score, high_level_score
