# -*- coding: utf-8 -*-
"""CMT class for converting model_call_history from AgentScope workflow to CMT object."""
import json
import copy
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.schema.trajectory import Reward
from games.games.avalon.utils import Parser

if TYPE_CHECKING:
    from agentevolver.schema.trajectory import Sample


class AgentscopeCMT(Linear_CMT):
    """
    CMT class for converting model_call_history from AgentScope workflow to CMT object.
    
    This class takes model_call_history (list of dicts with 'prompt' and 'response')
    and converts them into ExtendedMessage objects in full_context, which can then
    be tokenized using group_tokenize() to generate Sample objects.
    
    Each prompt-response pair in model_call_history will generate one Sample.
    """
    
    def __init__(self, config, tokenizer, model_call_history: List[Dict[str, Any]], 
                 reward: Optional[Reward] = None, data_id: str = "", 
                 rollout_id: str = "", task_id: str = ""):
        """
        Initialize AgentscopeCMT with model_call_history.
        
        Args:
            config: Configuration object containing environment and model settings.
            tokenizer: Tokenizer instance for processing text.
            model_call_history: List of dicts, each containing 'prompt' (list of messages) and 'response'.
            reward: Optional Reward object for the trajectory.
            data_id: Data ID for the trajectory.
            rollout_id: Rollout ID for the trajectory.
            task_id: Task ID for the trajectory.
        """
        super().__init__(config, tokenizer)
        
        # Set trajectory attributes
        self.data_id = data_id
        self.rollout_id = rollout_id
        self.task_id = task_id
        self.reward = reward if reward is not None else Reward()
        self.is_terminated = True
        
        # Store model_call_history for processing in group_tokenize
        self._build_full_context_from_history(model_call_history)
    
    def _build_sample_from_call_record(self, call_record: Dict[str, Any], minor_index_id: int):
        """
        Build a Sample from a single call record (prompt-response pair).
        
        Args:
            call_record: Dict with 'prompt' (list of messages) and 'response'.
            minor_index_id: Minor index ID for the sample.
        
        Returns:
            Sample: A Sample object for this prompt-response pair.
        """
        from agentevolver.schema.trajectory import Sample
        
        prompt_messages = call_record.get("prompt", [])
        response = call_record.get("response", "")
        
        # Ensure prompt_messages is a list
        if not isinstance(prompt_messages, list):
            # Fallback: try to parse if it's a string
            if isinstance(prompt_messages, str):
                try:
                    prompt_messages = json.loads(prompt_messages)
                except (json.JSONDecodeError, TypeError):
                    prompt_messages = [{"role": "user", "content": prompt_messages}]
            else:
                prompt_messages = [prompt_messages]
        
        # Build full_context for this single prompt-response pair
        # Use token_generator="manual" since we need to compute incremental tokens
        # based on the full conversation context, not individual messages
        full_context = []
        
        # Create ExtendedMessage objects for prompt messages
        # author="initialization" ensures prompt messages do NOT participate in training
        # (need_training=False, loss_mask will be all zeros)
        for msg in prompt_messages:
            if isinstance(msg, dict):
                ext_msg = ExtendedMessage(
                    author="initialization",  # Non-trainable: need_training=False
                    role=msg.get("role", "user"),
                    content=Parser.extract_text_from_content(msg.get("content", "")),
                    # content=str(msg.get_text_content()), # get text content in the text block
                    token_generator="manual",  # Use manual to compute context-based incremental tokens
                    tokenizer=self.tokenizer,
                )
                full_context.append(ext_msg)
        
        # Create ExtendedMessage for response
        # author="llm" ensures response message DOES participate in training
        # (need_training=True, loss_mask will be computed based on response tokens)
        ext_msg_response = ExtendedMessage(
            author="llm",  # Trainable: need_training=True
            role="assistant",
            content=str(response),
            token_generator="manual",  # Use manual to compute context-based incremental tokens
            tokenizer=self.tokenizer,
        )
        full_context.append(ext_msg_response)
        
        # Check if we have saved tokens from llm_chat_fn (for training consistency)
        saved_tokens = call_record.get("tokens", None)
        use_saved_tokens = saved_tokens is not None and len(saved_tokens) > 0
        
        # Compute token arrays for all messages
        if len(full_context) > 0:
            token_ids_acc = []
            messages_so_far = []
            
            for i, ext_msg in enumerate(full_context):
                # Build messages up to current point
                messages_so_far.append({
                    "role": ext_msg.role,
                    "content": ext_msg.content_for_future
                })
                
                # For response message, use saved tokens if available
                if ext_msg == ext_msg_response and use_saved_tokens:
                    # Use saved tokens from llm_chat_fn (similar to cmt_linear.py)
                    from agentevolver.module.context_manager.cmt_base import replace_token_ids
                    
                    # Calculate generation prompt tokens (difference between with/without generation prompt)
                    input_msg_ref = prompt_messages
                    generation_prompt_token, _ = self.get_inc(
                        self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=False),
                        self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=True),
                    )
                    
                    # Calculate completion token array (placeholder from text tokenization)
                    completion_token_arr, _ = self.get_inc(
                        self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False),
                        self.tokenizer.apply_chat_template(input_msg_ref + [{"role": "assistant", "content": str(response)}], tokenize=False),
                    )
                    
                    # Replace placeholder tokens with actual tokens from model output
                    # saved_tokens is already a list of token_ids (from AgentscopeModelWrapper)
                    if saved_tokens and isinstance(saved_tokens[0], int):
                        vllm_output_raw_token = saved_tokens
                    else:
                        # Fallback: convert token objects to token_ids
                        vllm_output_raw_token = [t.token_id if hasattr(t, 'token_id') else t for t in saved_tokens] if saved_tokens else []
                    final_token_arr = replace_token_ids(
                        place_holder=completion_token_arr,
                        replace_with=vllm_output_raw_token,
                        begin=generation_prompt_token,
                        end=[self.tokenizer.eos_token_id]
                    )
                    
                    # Set token_arr for response message
                    ext_msg_response.token_arr = final_token_arr
                    
                    # Update token_ids_acc for next iteration
                    # Build full context up to response
                    text_with_chat_template = self.tokenizer.apply_chat_template(
                        messages_so_far, tokenize=False
                    )
                    tokenizer_output = self.tokenizer(
                        text_with_chat_template, return_tensors="pt", padding=False
                    )
                    # For response, we need to use the actual tokens
                    # Calculate prompt tokens first
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_tokenizer_output = self.tokenizer(
                        prompt_text, return_tensors="pt", padding=False
                    )
                    prompt_token_ids = prompt_tokenizer_output["input_ids"][0].tolist()
                    # Combine prompt tokens with response tokens
                    token_ids_acc = prompt_token_ids + final_token_arr
                else:
                    # For prompt messages, use standard tokenization
                    text_with_chat_template = self.tokenizer.apply_chat_template(
                        messages_so_far, tokenize=False
                    )
                    tokenizer_output = self.tokenizer(
                        text_with_chat_template, return_tensors="pt", padding=False
                    )
                    input_ids = tokenizer_output["input_ids"][0].tolist()
                    
                    # Calculate incremental tokens (new tokens added by this message)
                    input_id_increment = input_ids[len(token_ids_acc):]
                    ext_msg.token_arr = input_id_increment
                    token_ids_acc = input_ids
        
        # Tokenize the steps
        cmt_tokenized = self.tokenize_steps(ext_steps=full_context)
        
        # Check if prompt is too long - if so, skip this sample
        prompt_len = len(cmt_tokenized["prompt_ids"])
        max_prompt_len = self.config.data.max_prompt_length
        if prompt_len > max_prompt_len:
            from loguru import logger
            logger.warning(
                f"Skipping sample (data_id={self.data_id}, minor_index_id={minor_index_id}): "
                f"prompt_ids length {prompt_len} exceeds max_prompt_len {max_prompt_len}"
            )
            return None
        
        # Create Sample
        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=minor_index_id,
            messages=self.to_role_content(full_context),
            input_ids=cmt_tokenized["input_ids"],
            prompt_ids=cmt_tokenized["prompt_ids"],
            response_ids=cmt_tokenized["response_ids"],
            attention_mask=cmt_tokenized["attention_mask"],
            prompt_attention_mask=cmt_tokenized["prompt_attention_mask"],
            response_attention_mask=cmt_tokenized["response_attention_mask"],
            loss_mask=cmt_tokenized["loss_mask"],
            prompt_loss_mask=cmt_tokenized["prompt_loss_mask"],
            response_loss_mask=cmt_tokenized["response_loss_mask"],
            position_ids=cmt_tokenized["position_ids"],
            prompt_position_ids=cmt_tokenized["prompt_position_ids"],
            response_position_ids=cmt_tokenized["response_position_ids"],
            reward_scores=self.reward.model_dump(),
            max_prompt_len=self.config.data.max_prompt_length,
            max_response_len=self.config.data.max_response_length,
            max_model_len=self.config.data.max_response_length + self.config.data.max_prompt_length,
        )
        sample.truncate_output_ids()
        
        return sample
    
    def group_tokenize(self):
        """
        Tokenize each prompt-response pair in model_call_history into a Sample.
        Samples with prompt length exceeding max_prompt_length will be skipped.
        
        Returns:
            List[Sample]: A list of Sample objects, one for each prompt-response pair.
        """
        sample_arr = []
        
        for minor_index_id, call_record in enumerate(self.model_call_history):
            sample = self._build_sample_from_call_record(call_record, minor_index_id)
            if sample is not None:  # Skip None samples (prompt too long)
                sample_arr.append(sample)
        
        return sample_arr
    
    def _build_full_context_from_history(self, model_call_history: List[Dict[str, Any]]):
        """
        Build full_context from model_call_history.
        Store each prompt-response pair separately for group_tokenize to process.
        
        Args:
            model_call_history: List of dicts with 'prompt' (list of messages) and 'response'.
        """
        # Store model_call_history for later use in group_tokenize
        self.model_call_history = model_call_history
        # full_context will be built per sample in group_tokenize
        self.full_context = []

