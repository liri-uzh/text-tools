"""
Extended test suite for multilingual preprocessing.
Tests both single-language and multilingual modes with edge cases.
"""

import pytest
from typing import List


# ============================================================================
# Original Tests (Backward Compatibility)
# ============================================================================


def test_phrasal_tokenizer_original():
    """Test original single-language tokenizer (German)"""
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(
        lang="de", mwes=["Dies academicus"], concat_token="_", lower=False
    )

    sentence = "Der Fotograf Jos Schmit hat für Dies academicus gearbeitet."
    tokenized = tokenizer.tokenize(sentence)

    assert tokenized == ["Fotograf", "Jos_Schmit", "Dies_academicus", "gearbeitet"]
    print("✓ Original German tokenizer test passed")


def test_mwe_parser_original():
    """Test original single-language MWE parser (German)"""
    from text_tools.preprocessing import MWEParser

    # Sample German texts for testing
    german_texts = [
        "Der Fotograf Jos Schmit hat für Dies academicus gearbeitet.",
        "Jos Schmit und Maria Müller arbeiten für Dies academicus.",
        "Das maschinelle Lernen ist ein wichtiger Bereich.",
        "Maschinelles Lernen wird in vielen Anwendungen eingesetzt.",
        "Die künstliche Intelligenz entwickelt sich rasant.",
        "Künstliche Intelligenz ist ein Teilgebiet der Informatik.",
    ]

    mwe_parser = MWEParser(
        lang="de",
        connector_words=["und", "oder", "für"],
        min_count=2,
        threshold=0.5,
    )

    mwe_parser.learn_phraser(german_texts)
    mwes = mwe_parser.extract_phrases()

    print(f"Found {len(mwes)} German multi-word expressions")
    print(f"German MWEs: {mwes}")

    assert len(mwes) > 0, "No multi-word expressions found"
    print("✓ Original German MWE parser test passed")


# ============================================================================
# Multilingual Tests
# ============================================================================


def test_multilingual_tokenizer_basic():
    """Test multilingual tokenizer with mixed-language input"""
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(
        lang="multi",
        mwes=["machine learning", "New York", "künstliche Intelligenz"],
        concat_token="_",
        lower=True,
    )

    test_cases = [
        {
            "text": "Machine learning is used in New York.",
            "expected_contains": ["machine_learning", "new_york"],
            "lang": "English",
        },
        {
            "text": "Das maschinelle Lernen ist wichtig.",
            "expected_contains": ["maschinelle", "lernen"],
            "lang": "German",
        },
        {
            "text": "L'intelligence artificielle est fascinante.",
            "expected_contains": ["l'intelligence", "artificielle"],
            "lang": "French",
        },
    ]

    for case in test_cases:
        tokens = tokenizer.tokenize(case["text"])
        print(f"\n{case['lang']}: {case['text']}")
        print(f"Tokens: {tokens}")

        for expected in case["expected_contains"]:
            assert expected in tokens, (
                f"Expected '{expected}' in tokens for {case['lang']}"
            )

    print("✓ Multilingual tokenizer basic test passed")


def test_multilingual_mwe_parser():
    """Test MWE detection across multiple languages"""
    from text_tools.preprocessing import MWEParser

    multilingual_texts = [
        # English
        "Machine learning is a subset of artificial intelligence.",
        "Artificial intelligence includes machine learning and deep learning.",
        "Deep learning is a powerful technique in machine learning.",
        # German
        "Das maschinelle Lernen ist ein Teilgebiet der künstlichen Intelligenz.",
        "Künstliche Intelligenz umfasst maschinelles Lernen und tiefes Lernen.",
        # French
        "L'apprentissage automatique est un sous-ensemble de l'intelligence artificielle.",
        "L'intelligence artificielle comprend l'apprentissage automatique.",
        # Spanish
        "El aprendizaje automático es parte de la inteligencia artificial.",
        "La inteligencia artificial incluye el aprendizaje automático.",
    ]

    parser = MWEParser(
        lang="multi",
        min_count=2,
        threshold=0.5,
    )

    parser.learn_phraser(multilingual_texts)
    mwes = parser.extract_phrases()

    print(f"\nFound {len(mwes)} multilingual MWEs")
    print(f"Sample MWEs: {mwes[:15]}")

    assert len(mwes) > 0, "No multilingual MWEs found"

    # Check that we found some language-specific phrases
    mwe_lower = [mwe.lower() for mwe in mwes]
    found_languages = []

    if any(
        "machine learning" in mwe or "artificial intelligence" in mwe
        for mwe in mwe_lower
    ):
        found_languages.append("English")
    if any("maschinelle" in mwe or "künstliche" in mwe for mwe in mwe_lower):
        found_languages.append("German")
    if any("apprentissage" in mwe or "intelligence" in mwe for mwe in mwe_lower):
        found_languages.append("French")

    print(f"Detected phrases from languages: {found_languages}")
    print("✓ Multilingual MWE parser test passed")


def test_multilingual_mixed_document():
    """Test handling of code-switched documents (multiple languages in one text)"""
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(
        lang="multi",
        concat_token="_",
        lower=True,
    )

    # Real-world scenario: academic paper with multiple languages
    mixed_text = """
    The concept of machine learning was introduced in the 20th century.
    Das maschinelle Lernen hat viele Anwendungen.
    L'apprentissage automatique est utilisé partout.
    """

    tokens = tokenizer.tokenize(mixed_text)
    print(f"\nMixed document tokens: {tokens}")

    # Should successfully tokenize without errors
    assert len(tokens) > 0, "Failed to tokenize mixed-language document"
    assert "machine" in tokens or "learning" in tokens, "English words not found"
    assert "maschinelle" in tokens or "lernen" in tokens, "German words not found"

    print("✓ Mixed document test passed")


# ============================================================================
# Edge Cases and Failure Modes
# ============================================================================


def test_empty_input():
    """Test handling of empty or whitespace-only input"""
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(lang="multi", lower=True)

    test_cases = [
        "",
        "   ",
        "\n\n\n",
        "\t\t",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        assert tokens == [], f"Expected empty list for '{repr(text)}', got {tokens}"

    print("✓ Empty input test passed")


def test_special_characters():
    """Test handling of special characters and unicode"""
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(
        lang="multi",
        concat_token="_",
        lower=True,
        keep_punct=False,
    )

    test_cases = [
        {
            "text": "Hello! How are you? I'm fine.",
            "should_not_contain": ["!", "?", "'", "."],
        },
        {
            "text": "Café résumé naïve",
            "expected_contains": ["café", "résumé", "naïve"],
        },
        {
            "text": "München über Zürich",
            "expected_contains": ["münchen", "zürich"],
        },
    ]

    for case in test_cases:
        tokens = tokenizer.tokenize(case["text"])
        print(f"\nText: {case['text']}")
        print(f"Tokens: {tokens}")

        if "should_not_contain" in case:
            for char in case["should_not_contain"]:
                assert char not in tokens, f"Punctuation '{char}' should be removed"

        if "expected_contains" in case:
            for expected in case["expected_contains"]:
                assert expected in tokens, f"Expected '{expected}' in tokens"

    print("✓ Special characters test passed")


def test_numbers_handling():
    """Test handling of numbers based on keep_num parameter"""
    from text_tools.preprocessing import PhrasalTokenizer

    # Test with keep_num=False (default)
    tokenizer_no_nums = PhrasalTokenizer(
        lang="multi",
        keep_num=False,
        lower=True,
    )

    text_with_nums = "In 2024, machine learning had 1000 applications."
    tokens = tokenizer_no_nums.tokenize(text_with_nums)
    print(f"\nWithout numbers: {tokens}")

    assert "2024" not in tokens, "Numbers should be filtered out"
    assert "1000" not in tokens, "Numbers should be filtered out"
    assert "machine" in tokens, "Words should be kept"

    # Test with keep_num=True
    tokenizer_with_nums = PhrasalTokenizer(
        lang="multi",
        keep_num=True,
        lower=True,
    )

    tokens = tokenizer_with_nums.tokenize(text_with_nums)
    print(f"With numbers: {tokens}")

    assert "2024" in tokens, "Numbers should be kept"
    assert "1000" in tokens, "Numbers should be kept"

    print("✓ Numbers handling test passed")


def test_stopwords_multilingual():
    """Test stopword handling across languages"""
    from text_tools.preprocessing import PhrasalTokenizer
    from text_tools.constants import STOP_WORDS

    stop_words = [*STOP_WORDS["en"], *STOP_WORDS["de"], *STOP_WORDS["fr"]]

    tokenizer = PhrasalTokenizer(
        lang="multi",
        stop_words=stop_words,
        keep_stopwords=False,
        lower=True,
    )

    test_cases = [
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "should_not_contain": ["the", "over"],  # English stopwords
            "should_contain": ["quick", "brown", "fox"],
        },
        {
            "text": "Der schnelle braune Fuchs springt über den faulen Hund",
            "should_not_contain": ["der", "den", "über"],  # German stopwords
            "should_contain": ["schnelle", "fuchs"],
        },
    ]

    print("\nTesting stopword removal:")
    for case in test_cases:
        tokens = tokenizer.tokenize(case["text"])
        print(f"Text: {case['text']}")
        print(f"Tokens: {tokens}")

        for word in case["should_not_contain"]:
            assert word not in tokens, f"Stopword '{word}' should be removed"

        for word in case["should_contain"]:
            assert word in tokens, f"Content word '{word}' should be kept"

    print("✓ Stopwords multilingual test passed")


def test_mwe_persistence():
    """Test saving and loading MWE models"""
    from text_tools.preprocessing import MWEParser
    import tempfile
    import os

    texts = [
        "Machine learning is important for artificial intelligence.",
        "Artificial intelligence uses machine learning techniques.",
        "Deep learning is a subset of machine learning.",
    ]

    parser = MWEParser(lang="multi", min_count=2, threshold=0.5)
    parser.learn_phraser(texts)
    mwes_original = parser.extract_phrases()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        parser.save_to_disk(tmp_path)

        # Create new parser and load
        new_parser = MWEParser(lang="multi")
        new_parser.load_from_disk(tmp_path)
        mwes_loaded = new_parser.extract_phrases()

        print(f"\nOriginal MWEs: {mwes_original}")
        print(f"Loaded MWEs: {mwes_loaded}")

        assert mwes_original == mwes_loaded, "Loaded MWEs don't match original"
        print("✓ MWE persistence test passed")

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_very_long_text():
    """Test handling of very long texts"""
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(lang="multi", lower=True)

    # Create a long text (simulating a document)
    long_text = " ".join(
        ["Machine learning is a powerful tool for data analysis." for _ in range(100)]
    )

    tokens = tokenizer.tokenize(long_text)

    assert len(tokens) > 0, "Failed to tokenize long text"
    assert "machine" in tokens, "Expected content not found in long text"

    print(f"✓ Long text test passed (tokenized {len(tokens)} tokens)")


def test_case_sensitivity():
    """Test case sensitivity handling"""
    from text_tools.preprocessing import PhrasalTokenizer

    # Test with lower=True
    tokenizer_lower = PhrasalTokenizer(
        lang="multi",
        mwes=["New York"],
        concat_token="_",
        lower=True,
    )

    # Test with lower=False
    tokenizer_no_lower = PhrasalTokenizer(
        lang="multi",
        mwes=["New York"],
        concat_token="_",
        lower=False,
    )

    text = "I visited New York and NEW YORK last year."

    tokens_lower = tokenizer_lower.tokenize(text)
    tokens_no_lower = tokenizer_no_lower.tokenize(text)

    print(f"\nLowercase: {tokens_lower}")
    print(f"Original case: {tokens_no_lower}")

    # Check lowercase version
    assert all(t.islower() or "_" in t for t in tokens_lower), (
        "All tokens should be lowercase"
    )

    # Check original case version has mixed case
    assert any(t[0].isupper() for t in tokens_no_lower if t and t[0].isalpha()), (
        "Should preserve uppercase"
    )

    print("✓ Case sensitivity test passed")


def test_url_email_handling():
    """Test handling of URLs and emails"""
    from text_tools.preprocessing import PhrasalTokenizer

    # Test with keep_url=False, keep_email=False (default)
    tokenizer_no_special = PhrasalTokenizer(
        lang="multi",
        keep_url=False,
        keep_email=False,
        lower=True,
    )

    text_with_special = """
    Visit https://example.com or email info@example.com for more information.
    You can also check www.example.org or contact support@company.de
    """

    tokens = tokenizer_no_special.tokenize(text_with_special)
    print(f"\nWithout URLs/emails: {tokens}")

    # URLs and emails should be filtered
    assert not any("http" in t or "@" in t or "www" in t for t in tokens), (
        "URLs and emails should be filtered"
    )
    assert "visit" in tokens, "Regular words should be kept"

    # Test with keep_url=True, keep_email=True
    tokenizer_with_special = PhrasalTokenizer(
        lang="multi",
        keep_url=True,
        keep_email=True,
        lower=True,
    )

    tokens = tokenizer_with_special.tokenize(text_with_special)
    print(f"With URLs/emails: {tokens}")

    # URLs and emails should be kept
    assert any("example.com" in t for t in tokens), "URLs should be kept"

    print("✓ URL/email handling test passed")


def test_insufficient_data_for_mwe():
    """Test MWE parser behavior with insufficient data"""
    from text_tools.preprocessing import MWEParser

    # Very few texts with no repetition
    sparse_texts = [
        "This is a unique sentence.",
        "Another completely different text.",
        "No patterns here at all.",
    ]

    parser = MWEParser(
        lang="multi",
        min_count=5,  # High threshold
        threshold=0.9,  # High threshold
    )

    parser.learn_phraser(sparse_texts)
    mwes = parser.extract_phrases()

    print(f"\nMWEs from sparse data: {mwes}")

    # Should handle gracefully (might find 0 or very few MWEs)
    assert isinstance(mwes, list), "Should return a list even with no MWEs"

    print("✓ Insufficient data test passed")


# ============================================================================
# Language Comparison Tests
# ============================================================================


def test_single_vs_multi_language_model_comparison():
    """Compare results between single-language and multilingual models"""
    from text_tools.preprocessing import PhrasalTokenizer

    german_text = (
        "Das maschinelle Lernen ist ein Teilgebiet der künstlichen Intelligenz."
    )

    # Single-language model
    tokenizer_de = PhrasalTokenizer(lang="de", lower=True)
    tokens_de = tokenizer_de.tokenize(german_text)

    # Multilingual model
    tokenizer_multi = PhrasalTokenizer(lang="multi", lower=True)
    tokens_multi = tokenizer_multi.tokenize(german_text)

    print(f"\nGerman model: {tokens_de}")
    print(f"Multi model: {tokens_multi}")

    # Both should produce similar results (though may differ slightly)
    assert len(tokens_de) > 0, "German model should produce tokens"
    assert len(tokens_multi) > 0, "Multi model should produce tokens"

    # Check overlap
    overlap = set(tokens_de) & set(tokens_multi)
    overlap_ratio = len(overlap) / max(len(tokens_de), len(tokens_multi))

    print(f"Overlap ratio: {overlap_ratio:.2%}")
    assert overlap_ratio > 0.5, "Should have significant overlap between models"

    print("✓ Model comparison test passed")


# ============================================================================
# Performance Sanity Checks
# ============================================================================


def test_batch_processing():
    """Test that batch processing works efficiently"""
    from text_tools.preprocessing import PhrasalTokenizer
    import time

    tokenizer = PhrasalTokenizer(lang="multi", lower=True)

    # Create a batch of texts
    texts = [
        "This is an English sentence about machine learning.",
        "Das ist ein deutscher Satz über maschinelles Lernen.",
        "Ceci est une phrase française sur l'apprentissage automatique.",
    ] * 10  # 30 texts total

    start_time = time.time()
    results = [tokenizer.tokenize(text) for text in texts]
    elapsed = time.time() - start_time

    print(f"\nProcessed {len(texts)} texts in {elapsed:.2f} seconds")
    print(f"Average: {elapsed / len(texts) * 1000:.2f}ms per text")

    assert len(results) == len(texts), "Should process all texts"
    assert all(len(tokens) > 0 for tokens in results), "All texts should produce tokens"

    print("✓ Batch processing test passed")


# ============================================================================
# Run All Tests
# ============================================================================


def run_all_tests():
    """Run all tests and report results"""
    tests = [
        # Original tests
        ("Original German Tokenizer", test_phrasal_tokenizer_original),
        ("Original German MWE Parser", test_mwe_parser_original),
        # Multilingual tests
        ("Multilingual Tokenizer Basic", test_multilingual_tokenizer_basic),
        ("Multilingual MWE Parser", test_multilingual_mwe_parser),
        ("Mixed Document", test_multilingual_mixed_document),
        # Edge cases
        ("Empty Input", test_empty_input),
        ("Special Characters", test_special_characters),
        ("Numbers Handling", test_numbers_handling),
        ("Stopwords Multilingual", test_stopwords_multilingual),
        ("MWE Persistence", test_mwe_persistence),
        ("Very Long Text", test_very_long_text),
        ("Case Sensitivity", test_case_sensitivity),
        ("URL/Email Handling", test_url_email_handling),
        ("Insufficient Data for MWE", test_insufficient_data_for_mwe),
        # Comparisons
        (
            "Single vs Multi Language Model",
            test_single_vs_multi_language_model_comparison,
        ),
        # Performance
        ("Batch Processing", test_batch_processing),
    ]

    print("=" * 70)
    print("RUNNING MULTILINGUAL PREPROCESSING TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        print(f"\n{'─' * 70}")
        print(f"Running: {test_name}")
        print("─" * 70)

        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_name, str(e)))
            print(f"✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_name, f"ERROR: {str(e)}"))
            print(f"✗ ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")

    if errors:
        print("\nFailed tests:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")

    print("=" * 70)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
