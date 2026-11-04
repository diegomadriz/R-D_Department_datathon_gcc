# Databricks notebook source
"""Optional local notebook wrapper for the authorized one-shot evaluation."""

from gcc_strength.lockbox import DEFAULT_OUTPUT, evaluate_lockbox, require_authorization

require_authorization()
result = evaluate_lockbox()
print(f"{result['status']} {DEFAULT_OUTPUT / 'result.json'}")
