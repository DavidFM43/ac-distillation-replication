"""Agent prompts, adapted from mini-swe-agent's SWE-bench config (MIT License,
https://github.com/SWE-agent/mini-swe-agent, config/benchmarks/swebench.yaml).

Changes from the original:
- Submission is a `task_complete` tool call instead of `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Grading runs the tests on the sandbox's final state, so no patch file is needed.
- The response example refers to our two tools.
- No turn limit is mentioned.
"""

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."

INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</IMPORTANT>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE tool call. You can make MULTIPLE bash tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, call the `task_complete` tool.
Your changes to the files in /testbed are what will be evaluated.

<IMPORTANT>
Before submitting, remove any files you created that are not part of the fix
(reproduction scripts, helper scripts, scratch files).
</IMPORTANT>

<CRITICAL>
- You CANNOT continue working (reading, editing, testing) in any way on this task after calling `task_complete`.
</CRITICAL>
</instructions>"""


def observation(returncode: int, output: str, max_chars: int = 10000) -> str:
    """Tool output formatting, as in mini-swe-agent: long outputs keep head and tail."""
    if len(output) < max_chars:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    half = max_chars // 2
    return (
        f"<returncode>{returncode}</returncode>\n"
        "<warning>\n"
        "The output of your last command was too long.\n"
        "Please try a different command that produces less output.\n"
        "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
        "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
        "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
        "</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{len(output) - 2 * half} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )
