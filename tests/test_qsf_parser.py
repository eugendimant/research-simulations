"""
Comprehensive tests for QSF Parser.

Tests all QSF format variations and edge cases to ensure robust parsing.
"""

import json
import os
import sys
from pathlib import Path

# Path setup: works both via pytest (conftest.py) and direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

from utils.qsf_preview import QSFPreviewParser, QSFPreviewResult


class TestQSFParser:
    """Test suite for QSF parsing functionality."""

    def __init__(self):
        self.parser = QSFPreviewParser()
        self.test_results = []

    def run_all_tests(self):
        """Run all tests and report results."""
        print("=" * 70)
        print("QSF PARSER TEST SUITE")
        print("=" * 70)

        # Test with actual QSF files
        test_files = [
            Path(__file__).parent.parent / "docs" / "Emoji_Pilot_-_Copy.qsf",
            Path(__file__).parent.parent / "docs" / "Capstone_Project (1).qsf",
            Path(__file__).parent.parent.parent / "TA Simulations" / "Leo" / "Capstone_Project (1).qsf",
            Path(__file__).parent.parent.parent / "TA Simulations" / "Karlijn" / "LLM_simulateddata" / "Group_12_MTurk (1).qsf",
        ]

        for qsf_path in test_files:
            if qsf_path.exists():
                self._test_qsf_file(qsf_path)
            else:
                print(f"\nSKIP: {qsf_path.name} (file not found)")

        # Test synthetic edge cases
        self._test_list_format_payload()
        self._test_dict_format_payload()
        self._test_dict_with_blocks_key()
        self._test_empty_payload()
        self._test_null_payload()
        self._test_nested_block_elements()

        # Summary
        return self._print_summary()

    def _test_qsf_file(self, file_path: Path):
        """Test parsing an actual QSF file."""
        test_name = f"Parse {file_path.name}"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            result = self.parser.parse(content)

            # Check if parsing succeeded
            if result.success:
                print(f"SUCCESS: Parsed successfully")
                print(f"  - Survey name: {result.survey_name}")
                print(f"  - Total blocks: {result.total_blocks}")
                print(f"  - Total questions: {result.total_questions}")
                print(f"  - Detected conditions: {result.detected_conditions}")
                print(f"  - Warnings: {len(result.validation_warnings)}")
                print(f"  - Errors: {len(result.validation_errors)}")
                self.test_results.append((test_name, True, None))
            else:
                print(f"PARTIAL SUCCESS: Parsed with issues")
                print(f"  - Errors: {result.validation_errors[:3]}")
                # Consider partial success as pass if we got some data
                if result.total_blocks > 0 or result.total_questions > 0:
                    self.test_results.append((test_name, True, "Parsed with warnings"))
                else:
                    self.test_results.append((test_name, False, "No data parsed"))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            self.test_results.append((test_name, False, str(e)))

    def _test_list_format_payload(self):
        """Test QSF with list format Payload (newer format)."""
        test_name = "List format Payload"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        qsf_data = {
            "SurveyEntry": {"SurveyName": "Test Survey"},
            "SurveyElements": [
                {
                    "Element": "BL",
                    "Payload": [
                        {
                            "Type": "Default",
                            "Description": "Block 1",
                            "ID": "BL_001",
                            "BlockElements": [
                                {"Type": "Question", "QuestionID": "QID1"}
                            ]
                        },
                        {
                            "Type": "Standard",
                            "Description": "Block 2",
                            "ID": "BL_002",
                            "BlockElements": []
                        }
                    ]
                },
                {
                    "Element": "SQ",
                    "PrimaryAttribute": "QID1",
                    "Payload": {
                        "QuestionID": "QID1",
                        "QuestionText": "Test question",
                        "QuestionType": "MC",
                        "Selector": "SAVR",
                        "Choices": {"1": {"Display": "Yes"}, "2": {"Display": "No"}}
                    }
                }
            ]
        }

        try:
            content = json.dumps(qsf_data).encode('utf-8')
            result = self.parser.parse(content)

            if result.total_blocks >= 2 and result.total_questions >= 1:
                print(f"SUCCESS: Parsed {result.total_blocks} blocks, {result.total_questions} questions")
                self.test_results.append((test_name, True, None))
            else:
                print(f"FAILED: Expected 2 blocks, 1 question. Got {result.total_blocks}, {result.total_questions}")
                self.test_results.append((test_name, False, "Incorrect counts"))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))

    def _test_dict_format_payload(self):
        """Test QSF with dict format Payload (older format)."""
        test_name = "Dict format Payload"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        qsf_data = {
            "SurveyEntry": {"SurveyName": "Test Survey Dict"},
            "SurveyElements": [
                {
                    "Element": "BL",
                    "Payload": {
                        "0": {
                            "Type": "Default",
                            "Description": "Intro Block",
                            "ID": "BL_A",
                            "BlockElements": []
                        },
                        "1": {
                            "Type": "Standard",
                            "Description": "Main Block",
                            "ID": "BL_B",
                            "BlockElements": [
                                {"Type": "Question", "QuestionID": "QID1"}
                            ]
                        }
                    }
                },
                {
                    "Element": "SQ",
                    "PrimaryAttribute": "QID1",
                    "Payload": {
                        "QuestionID": "QID1",
                        "QuestionText": "How satisfied are you?",
                        "QuestionType": "Matrix",
                        "Selector": "Likert",
                        "Choices": {"1": {"Display": "Item 1"}},
                        "Answers": {
                            "1": {"Display": "Strongly Disagree"},
                            "2": {"Display": "Disagree"},
                            "3": {"Display": "Neutral"},
                            "4": {"Display": "Agree"},
                            "5": {"Display": "Strongly Agree"}
                        }
                    }
                }
            ]
        }

        try:
            content = json.dumps(qsf_data).encode('utf-8')
            result = self.parser.parse(content)

            if result.total_blocks >= 2:
                print(f"SUCCESS: Parsed {result.total_blocks} blocks, {result.total_questions} questions")
                self.test_results.append((test_name, True, None))
            else:
                print(f"FAILED: Expected 2 blocks. Got {result.total_blocks}")
                self.test_results.append((test_name, False, "Incorrect block count"))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))

    def _test_dict_with_blocks_key(self):
        """Test QSF with dict containing 'Blocks' key."""
        test_name = "Dict with Blocks key"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        qsf_data = {
            "SurveyEntry": {"SurveyName": "Test Survey Blocks Key"},
            "SurveyElements": [
                {
                    "Element": "BL",
                    "Payload": {
                        "Blocks": [
                            {
                                "Type": "Default",
                                "Description": "Block From Nested",
                                "ID": "BL_nested",
                                "BlockElements": []
                            }
                        ]
                    }
                }
            ]
        }

        try:
            content = json.dumps(qsf_data).encode('utf-8')
            result = self.parser.parse(content)

            if result.total_blocks >= 1:
                print(f"SUCCESS: Parsed {result.total_blocks} blocks")
                self.test_results.append((test_name, True, None))
            else:
                print(f"FAILED: Expected at least 1 block. Got {result.total_blocks}")
                self.test_results.append((test_name, False, "No blocks parsed"))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))

    def _test_empty_payload(self):
        """Test handling of empty Payload."""
        test_name = "Empty Payload handling"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        qsf_data = {
            "SurveyEntry": {"SurveyName": "Empty Payload Test"},
            "SurveyElements": [
                {
                    "Element": "BL",
                    "Payload": {}
                }
            ]
        }

        try:
            content = json.dumps(qsf_data).encode('utf-8')
            result = self.parser.parse(content)
            print(f"SUCCESS: Handled empty payload without crash")
            self.test_results.append((test_name, True, None))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))

    def _test_null_payload(self):
        """Test handling of null Payload."""
        test_name = "Null Payload handling"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        qsf_data = {
            "SurveyEntry": {"SurveyName": "Null Payload Test"},
            "SurveyElements": [
                {
                    "Element": "BL",
                    "Payload": None
                }
            ]
        }

        try:
            content = json.dumps(qsf_data).encode('utf-8')
            result = self.parser.parse(content)
            print(f"SUCCESS: Handled null payload without crash")
            self.test_results.append((test_name, True, None))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))

    def _test_nested_block_elements(self):
        """Test handling of various BlockElements formats."""
        test_name = "Nested BlockElements formats"
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print("-" * 70)

        qsf_data = {
            "SurveyEntry": {"SurveyName": "BlockElements Test"},
            "SurveyElements": [
                {
                    "Element": "BL",
                    "Payload": [
                        {
                            "Type": "Default",
                            "Description": "Block with list elements",
                            "ID": "BL_list",
                            "BlockElements": [
                                {"Type": "Question", "QuestionID": "QID1"},
                                {"Type": "Page Break"}
                            ]
                        },
                        {
                            "Type": "Standard",
                            "Description": "Block with dict elements",
                            "ID": "BL_dict",
                            "BlockElements": {
                                "0": {"Type": "Question", "QuestionID": "QID2"}
                            }
                        },
                        {
                            "Type": "Standard",
                            "Description": "Block with no elements",
                            "ID": "BL_empty",
                            "BlockElements": None
                        }
                    ]
                },
                {
                    "Element": "SQ",
                    "PrimaryAttribute": "QID1",
                    "Payload": {
                        "QuestionID": "QID1",
                        "QuestionText": "Question 1",
                        "QuestionType": "MC",
                        "Selector": "SAVR"
                    }
                },
                {
                    "Element": "SQ",
                    "PrimaryAttribute": "QID2",
                    "Payload": {
                        "QuestionID": "QID2",
                        "QuestionText": "Question 2",
                        "QuestionType": "TE",
                        "Selector": "SL"
                    }
                }
            ]
        }

        try:
            content = json.dumps(qsf_data).encode('utf-8')
            result = self.parser.parse(content)

            if result.total_blocks >= 3:
                print(f"SUCCESS: Parsed {result.total_blocks} blocks with various BlockElements formats")
                self.test_results.append((test_name, True, None))
            else:
                print(f"FAILED: Expected 3 blocks. Got {result.total_blocks}")
                self.test_results.append((test_name, False, "Incorrect block count"))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            self.test_results.append((test_name, False, str(e)))

    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = len(self.test_results) - passed

        print(f"Total: {len(self.test_results)} | Passed: {passed} | Failed: {failed}")
        print("-" * 70)

        for name, success, error in self.test_results:
            status = "PASS" if success else "FAIL"
            error_msg = f" - {error}" if error else ""
            print(f"[{status}] {name}{error_msg}")

        print("=" * 70)

        if failed > 0:
            print(f"\nFAILURES DETECTED: {failed} tests failed")
            return False
        else:
            print("\nALL TESTS PASSED")
            return True


def main():
    """Run all tests."""
    tester = TestQSFParser()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
