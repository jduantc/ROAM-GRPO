"""Logical soundness prompt: assess the solution's reasoning.

Catches setup errors (misread quantities), wrong formulas, unit/dimension errors,
arithmetic slips, and conclusions that don't follow from the steps shown.

This module is imported by routed_offmode.py at training time.
"""
VALIDITY = (
    "Audit the reasoning in the following solution. Be concise -- at most 5 short lines.\n\n"
    "PROBLEM:\n{problem}\n\n"
    "SOLUTION TO AUDIT:\n{b}\n\n"
    "Check specifically for:\n"
    "- a quantity in the problem read as the wrong kind of thing (a distance used as a time, etc.)\n"
    "- a wrong or misapplied formula\n"
    "- unit / dimension errors (multiplying where it should divide)\n"
    "- arithmetic slips\n"
    "- a conclusion that does not follow from the steps shown\n\n"
    "Do not solve the problem yourself. Judge only whether the steps shown are logically and "
    "dimensionally valid.\n\n"
    "End with exactly this line and nothing after it:\n"
    "VALIDITY: sound   (or)   VALIDITY: unsound\n"
)
