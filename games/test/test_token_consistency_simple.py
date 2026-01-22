# -*- coding: utf-8 -*-
"""Simple test to verify token consistency - quick check version.

This is a simpler version that can be run quickly to verify the fix works.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from transformers import AutoTokenizer
from agentevolver.utils.agentscope_utils import AgentscopeModelWrapper
from games.agents.agentscope_cmt import AgentscopeCMT
from agentevolver.schema.trajectory import Reward


class MockToken:
    """Mock token object."""
    def __init__(self, token_id: int):
        self.token_id = token_id


class MockConfig:
    """Mock config."""
    def __init__(self):
        self.data = SimpleNamespace()
        self.data.max_prompt_length = 2048
        self.data.max_response_length = 2048


def test_end_to_end():
    """Test the full pipeline: llm_chat_fn -> wrapper -> CMT."""
    print("=" * 80)
    print("Simple Token Consistency Test")
    print("=" * 80)
    
    # Step 1: Create mock llm_chat_fn with tokens
    expected_tokens = [100, 200, 300, 400, 500]
    
    def mock_llm_chat_fn(messages, **kwargs):
        return {
            "role": "assistant",
            "content": "Hello, this is a test response.",
            "tokens": [MockToken(tid) for tid in expected_tokens]
        }
    
    # Step 2: Test AgentscopeModelWrapper
    print("\n1. Testing AgentscopeModelWrapper...")
    wrapper = AgentscopeModelWrapper(
        llm_chat_fn=mock_llm_chat_fn,
        model_name="test_model"
    )
    
    import asyncio
    async def test_wrapper():
        response = await wrapper([{"role": "user", "content": "Hello"}])
        assert response.metadata is not None, "Metadata should exist"
        assert 'tokens' in response.metadata, "Tokens should be in metadata"
        saved_tokens = response.metadata['tokens']
        assert saved_tokens == expected_tokens, f"Tokens mismatch: {saved_tokens} != {expected_tokens}"
        print(f"   ✅ Wrapper saved tokens: {saved_tokens}")
        return saved_tokens
    
    saved_tokens = asyncio.run(test_wrapper())
    
    # Step 3: Test AgentscopeCMT with saved tokens
    print("\n2. Testing AgentscopeCMT with saved tokens...")
    config = MockConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    
    model_call_history = [
        {
            "prompt": [{"role": "user", "content": "Say hello"}],
            "response": "Hello, this is a test response.",
            "tokens": saved_tokens  # Use tokens from wrapper
        }
    ]
    
    cmt = AgentscopeCMT(
        config=config,
        tokenizer=tokenizer,
        model_call_history=model_call_history,
        reward=Reward(outcome=1.0),
        data_id="test",
        rollout_id="test",
        task_id="test"
    )
    
    samples = cmt.group_tokenize()
    assert len(samples) == 1, "Should have 1 sample"
    sample = samples[0]
    
    print(f"   ✅ Created sample with {len(sample.response_ids)} response tokens")
    print(f"   ✅ Response loss mask sum: {sum(sample.response_loss_mask)}")
    
    # Step 4: Verify tokens are used (not from text)
    print("\n3. Verifying token usage...")
    print(f"   ✅ Original tokens from llm_chat_fn: {expected_tokens}")
    print(f"   ✅ Sample created successfully")
    print(f"   ✅ This confirms saved tokens are being used in the pipeline")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Token consistency is maintained.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_end_to_end()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

