"""Distinctness judge prompt: scores how similar two solutions' methods are.

Given two solutions to the same problem, the LLM judge lists each solution's
key mathematical moves, compares them, and outputs a SCORE (0-100).
100 = certainly the same method, 0 = certainly different.

Stage 2 validity (is the reasoning sound?) is handled by validity_prompt.py.

This module is imported by routed_offmode.py at training time.
"""

DISTINCTNESS = (
    "You are given two solutions to the same mathematics problem.\n\n"
    "METHOD COMPARISON.\n"
    "STEPS A: list solution A's key mathematical moves as a numbered list of 3-6 short phrases. Name "
    "the OPERATION performed (e.g. 'set up coordinates', 'apply Vieta's formulas', 'complementary "
    "counting'), not the numbers involved.\n"
    "STEPS B: do the same for solution B.\n"
    "VERDICT: compare the two lists. They are the SAME method if they reach the answer by the same "
    "route, even if worded differently, of very different length, or containing arithmetic slips. "
    "They are DIFFERENT only if at least one step differs in KIND: a different theorem or identity "
    "invoked, a different representation of the object, casework versus a closed form, or algebraic "
    "versus geometric versus counting versus brute-force enumeration.\n\n"
    "Then output exactly this line and nothing after it:\n"
    "SCORE: <integer 0-100>   (100 = certainly the SAME method as A, 50 = unsure, 0 = certainly "
    "DIFFERENT methods)\n\n"
    "PROBLEM:\n{problem}\n\nSOLUTION A:\n{a}\n\nSOLUTION B:\n{b}\n")
