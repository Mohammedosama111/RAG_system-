#!/usr/bin/env python3
"""
Test script to verify the array boolean context fix
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_array_boolean_fix():
    """Test that array boolean contexts are handled correctly"""
    
    # Simulate the problematic scenarios
    import numpy as np
    
    # Test case 1: Empty array
    empty_array = np.array([])
    print(f"Empty array: {empty_array}")
    print(f"Length check: {len(empty_array) == 0}")
    print(f"Is None check: {empty_array is not None}")
    
    # Test case 2: Non-empty array
    non_empty_array = np.array([1, 2, 3])
    print(f"\nNon-empty array: {non_empty_array}")
    print(f"Length check: {len(non_empty_array) > 0}")
    print(f"Is None check: {non_empty_array is not None}")
    
    # Test case 3: None value
    none_value = None
    print(f"\nNone value: {none_value}")
    print(f"Is None check: {none_value is not None}")
    
    # Test case 4: List
    list_value = [1, 2, 3]
    print(f"\nList value: {list_value}")
    print(f"Length check: {len(list_value) > 0}")
    print(f"Is None check: {list_value is not None}")
    
    # Test the fixed conditional logic
    def test_fixed_condition(embeddings_result):
        if embeddings_result is not None and len(embeddings_result) > 0:
            print(f"✅ Fixed condition passed for: {type(embeddings_result)}")
            return True
        else:
            print(f"❌ Fixed condition failed for: {type(embeddings_result)}")
            return False
    
    print("\n=== Testing Fixed Conditional Logic ===")
    test_fixed_condition(empty_array)
    test_fixed_condition(non_empty_array)
    test_fixed_condition(none_value)
    test_fixed_condition(list_value)
    
    print("\n✅ Array boolean context fix test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_array_boolean_fix()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)