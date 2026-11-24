"""Unit tests for source normalization, Tree-sitter parsing, and diagnostic extraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.normalizer import NUM_STATES, normalize_source
from src.preprocessing.parser import ParseResult, parse_to_state_sequence

JAVA_SAMPLE = """\
package contest.round1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {
    static class FastScanner {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        String next() throws Exception {
            while (st == null || !st.hasMoreElements()) {
                st = new StringTokenizer(br.readLine());
            }
            return st.nextToken();
        }
    }

    public static int solve(int[] nums) {
        int total = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                total += nums[i] * 2;
            } else {
                total -= nums[i];
            }
        }
        while (total > 100) {
            total /= 2;
        }
        try {
            return total;
        } catch (RuntimeException ex) {
            throw ex;
        }
    }
}
"""

PYTHON_SAMPLE = """\
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)
input = sys.stdin.readline

class Solver:
    def solve(self, nums: list[int]) -> int:
        total = 0
        for idx, val in enumerate(nums):
            if val > 0:
                total += val * 2
            elif val < 0:
                total -= val
            else:
                continue
        while total > 100:
            total //= 2
        try:
            return total
        except ValueError as err:
            raise RuntimeError(str(err))
"""

CPP_SAMPLE = """\
#include <iostream>
#include <vector>
using namespace std;

class Solver {
public:
    int solve(const vector<int>& nums) {
        ios_base::sync_with_stdio(false);
        cin.tie(nullptr);
        int total = 0;
        for (size_t i = 0; i < nums.size(); ++i) {
            if (nums[i] > 0) {
                total += nums[i] * 2;
            } else {
                total -= nums[i];
            }
        }
        while (total > 100) {
            total /= 2;
        }
        try {
            return total;
        } catch (const exception& ex) {
            throw;
        }
    }
};
"""


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("java", JAVA_SAMPLE),
        ("python", PYTHON_SAMPLE),
        ("cpp", CPP_SAMPLE),
    ],
)
def test_parse_supported_languages_and_low_other_ratio(language: str, source: str):
    result = parse_to_state_sequence(source, language=language, strip_boilerplate=True)

    assert isinstance(result, ParseResult)
    assert result.states.dtype == np.int32
    assert result.states.ndim == 1
    assert result.token_count == result.states.size
    assert result.token_count > 10
    assert not result.has_error
    assert result.error_count == 0
    assert int(result.states.min()) >= 0
    assert int(result.states.max()) < NUM_STATES
    assert result.other_ratio < 0.25


def test_unsupported_language_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported language"):
        parse_to_state_sequence("fn main() {}", language="rust")

    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_source("fn main() {}", language="rust")


def test_nbsp_and_line_ending_normalization():
    dirty_java = "public\xa0class\xa0Demo\r\n{\r\n\xa0\xa0int\xa0x\xa0=\xa01;\r\n}\r\n"

    normalized = normalize_source(dirty_java, language="java", strip_boilerplate=False)
    assert "\xa0" not in normalized
    assert "\r" not in normalized

    result = parse_to_state_sequence(dirty_java, language="java", strip_boilerplate=True)
    assert not result.has_error
    assert result.token_count > 0


def test_boilerplate_stripping_reduces_templates_without_breaking_ast():
    java_norm = normalize_source(JAVA_SAMPLE, language="java", strip_boilerplate=True)
    assert "package contest.round1;" not in java_norm
    assert "import java.io.BufferedReader;" not in java_norm
    assert "class FastScanner" not in java_norm
    assert "public static int solve" in java_norm
    assert len(java_norm) < len(JAVA_SAMPLE)

    py_norm = normalize_source(PYTHON_SAMPLE, language="python", strip_boilerplate=True)
    assert "import sys" not in py_norm
    assert "from collections import defaultdict" not in py_norm
    assert "setrecursionlimit" not in py_norm
    assert "sys.stdin.readline" not in py_norm
    assert "class Solver:" in py_norm

    cpp_norm = normalize_source(CPP_SAMPLE, language="cpp", strip_boilerplate=True)
    assert "#include" not in cpp_norm
    assert "using namespace std;" not in cpp_norm
    assert "sync_with_stdio" not in cpp_norm
    assert "cin.tie" not in cpp_norm
    assert "class Solver" in cpp_norm


def test_partial_syntax_error_recovery():
    broken_java = """\
    public class Broken {
        public int compute(int n) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                sum += i;
            }
            if (sum > ) {
                return sum;
            }
            return 0;
        }
    }
    """
    result = parse_to_state_sequence(broken_java, language="java", strip_boilerplate=True)
    assert result.has_error is True
    assert result.error_count >= 1
    assert result.token_count > 5
    assert result.states.dtype == np.int32
    assert int(result.states.min()) >= 0
    assert int(result.states.max()) < NUM_STATES
