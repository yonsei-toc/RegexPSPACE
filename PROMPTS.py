PROMPTS = {"minimization_zero_prompt" : """You are an expert in formal regular expressions, commonly referred to as regex.

The formal regex must follow these rules:
- Allowed operations: concatenation, union (`+`), Kleene star (`*`), and option (`?`).
- Concatenation is implicit (no symbol is written).
- Parentheses `()` specify precedence.
- Do not use practical regex notations such as `|` for union or `+` for repetition.

Your task is to minimize a given formal regex over the fixed alphabet {`a`, `b`, `c`, `d`}. The minimized regex must be functionally equivalent to the input and have the smallest total number of symbols, where both characters (`a`, `b`, `c`, `d`) and operations (`+`, `*`, `?`), and concatenations are counted, but parentheses are not.

Enclose your final answer within `\\boxed{}`.

Minimize the following regex:

Input Regex: $regex
Output: """,



"equivalence_zero_prompt" : """You are an expert in formal regular expressions, commonly referred to as regex.

The formal regex must follow these rules:
- Allowed operations: concatenation, union (`+`), Kleene star (`*`), and option (`?`).
- Concatenation is implicit (no symbol is written).
- Parentheses `()` specify precedence.
- Do not use practical regex notations such as `|` for union or `+` for repetition.

Your task is to determine whether two given formal regexes over the fixed alphabet {`a`, `b`, `c`, `d`} are equivalent. You must output either True or False.

Enclose your final answer within `\\boxed{}`.

Determine the equivalence of the following regexes:

Input Regex 1: $regex1
Input Regex 2: $regex2
Output: """,



"minimization_five_prompt" : """You are an expert in formal regular expressions, commonly referred to as regex.

The formal regex must follow these rules:
- Allowed operations: concatenation, union (`+`), Kleene star (`*`), and option (`?`).
- Concatenation is implicit (no symbol is written).
- Parentheses `()` specify precedence.
- Do not use practical regex notations such as `|` for union or `+` for repetition.

Your task is to minimize a given formal regex over the fixed alphabet {`a`, `b`, `c`, `d`}. The minimized regex must be functionally equivalent to the input and have the smallest total number of symbols, where both characters (`a`, `b`, `c`, `d`) and operations (`+`, `*`, `?`), and concatenations are counted, but parentheses are not.

Enclose your final answer within `\\boxed{}`.

Below are five input-output examples:

[Example 1]
Input Regex: $example1_input_regex
Output: $example1_output

[Example 2]
Input Regex: $example2_input_regex
Output: $example2_output

[Example 3]
Input Regex: $example3_input_regex
Output: $example3_output

[Example 4]
Input Regex: $example4_input_regex
Output: $example4_output

[Example 5]
Input Regex: $example5_input_regex
Output: $example5_output

Minimize the following regex:

Input Regex: $regex
Output: """,



"equivalence_five_prompt" : """You are an expert in formal regular expressions, commonly referred to as regex.

The formal regex must follow these rules:
- Allowed operations: concatenation, union (`+`), Kleene star (`*`), and option (`?`).
- Concatenation is implicit (no symbol is written).
- Parentheses `()` specify precedence.
- Do not use practical regex notations such as `|` for union or `+` for repetition.

Your task is to determine whether two given formal regexes over the fixed alphabet {`a`, `b`, `c`, `d`} are equivalent. You must output either True or False.

Enclose your final answer within `\\boxed{}`.

Below are five input-output examples:

[Example 1]
Input Regex 1: $example1_input_regex1
Input Regex 2: $example1_input_regex2
Output: $example1_output

[Example 2]
Input Regex 1: $example2_input_regex1
Input Regex 2: $example2_input_regex2
Output: $example2_output

[Example 3]
Input Regex 1: $example3_input_regex1
Input Regex 2: $example3_input_regex2
Output: $example3_output

[Example 4]
Input Regex 1: $example4_input_regex1
Input Regex 2: $example4_input_regex2
Output: $example4_output

[Example 5]
Input Regex 1: $example5_input_regex1
Input Regex 2: $example5_input_regex2
Output: $example5_output

Determine the equivalence of the following regexes:

Input Regex 1: $regex1
Input Regex 2: $regex2
Output: """
}