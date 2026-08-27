"""Which tests in here cannot fail?

The defect this repository keeps finding is not the crash. It is the check
that runs, exits 0, and has no effect: `compare.py` reading a shape `speed.py`
had stopped writing, so every sweep printed a table of dashes; `-ot
"shard_[0-9]+=CPU"` matching nothing and reading exactly like a placement that
had already happened; `llama-probe.timer` linked but never enabled, so the
watchdog had never once fired; a simulated-clone command whose exclude list
excluded nothing; a `# shellcheck source=` directive voided by the prose after
it on the same line.

A test can have that shape too, and then it is worse than no test, because it
is also a claim. On 27.08.2026 this file was written by sweeping the suite for
it, and it found one that was live:
`test_scripts.py::TestUserUnit::test_nothing_restores_slots_at_start` looped
over `ExecStartPost` and asserted inside the loop. The directive had been
removed from the unit, so the body never ran. The test guarding against the
boot restore that poisoned a freshly started server had been passing without
reading anything.

WHAT IS FLAGGED, and only this: a test whose assertions ALL live inside a
`for` loop whose iterable cannot be shown to be non-empty. A loop over a
literal, or over a constant that is a non-empty literal, is fine — it cannot
iterate zero times. A loop over a glob, a helper call or a parsed collection
can, and then a green test means "nothing was checked" and "everything
checked out" indistinguishably.

THE CURE IS NOT TO STOP LOOPING. It is a positive control: assert the
collection is non-empty first, or assert something outside the loop that
proves the reader found its subject. `test_nothing_restores_slots_at_start`
now asserts that the parser finds `ExecStart` before concluding anything about
`ExecStartPost`.

The list below is what was in the suite when the sweep was written. It is
named rather than remembered, for the reason the same pattern is named in
tests/test_localenv.py and tests/test_models.py: an exception nobody can see
is how a temporary state becomes permanent. It can only shrink — a name that
no longer applies fails the second test.
"""
import ast
import glob
import os
import unittest

import common

TESTS = str(common.REPO / "tests")

# Grandfathered on 27.08.2026. Every one of these loops over something this
# analysis cannot prove non-empty — a glob, a registry, a parsed listing. None
# of them is known to be vacuous today; all of them WOULD be if their subject
# ever went missing, which is the state that made this file worth writing.
#
# Adding to this list is a decision to accept that. Prefer a positive control.
KNOWN = {
    "test_defects.py::TestTheRegistryItself.test_shows_as_is_one_of_the_four",
    "test_defects.py::TestTheRegistryItself.test_every_named_suite_exists",
    "test_defects.py::TestTheRegistryItself.test_upstream_entries_are_urls",
    "test_defects.py::TestSilenceIsNotSafety.test_an_argument_check_is_unknown_with_nothing_running",
    "test_gateway.py::TestModelListing.test_aliases_inherit_what_the_server_reports",
    "test_hardware.py::TestItIdentifiesTheGpuTwice.test_every_id_in_the_table_says_where_it_was_seen",
    "test_hardware.py::TestTheDefectRegistryFinallyReadsItsOwnField.test_a_foreign_gpu_drops_them",
    "test_hardware.py::TestTheDefectRegistryFinallyReadsItsOwnField.test_the_target_gpu_keeps_them",
    "test_hardware.py::TestTheDefectRegistryFinallyReadsItsOwnField.test_an_unknown_gpu_keeps_them_too",
    "test_localenv.py::TestNothingInTheRepoNamesOneMachine.test_no_variants_file_names_one_machine_s_builds",
    "test_localenv.py::TestNothingInTheRepoNamesOneMachine.test_the_profiles_rule_still_holds",
    "test_models.py::TestRegistry.test_every_profile_declares_its_metadata",
    "test_models.py::TestRegistry.test_metadata_lookup_does_not_confuse_two_variables_with_a_prefix",
    "test_models.py::TestRegistry.test_every_profile_carries_llama_args",
    "test_models.py::TestRegistry.test_the_alias_is_the_profile_name",
    "test_models.py::TestArgsReader.test_every_caller_gets_the_same_answer",
    "test_models.py::TestNothingStartsAModelWithoutAskingFirst.test_neither_guard_is_prefixed_with_a_dash",
    "test_models.py::TestSwitchPreflight.test_every_profile_says_where_its_model_comes_from",
    "test_speed.py::TestPayloads.test_a_shallow_depth_never_produces_an_empty_prompt",
    "test_sweep.py::TestVariantsFile.test_the_json_kwargs_are_valid_json_in_every_variant",
    "test_systemunit.py::TestTheDerivationIsComplete.test_no_user_unit_instance_name_survives",
}


def _is_assert(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr.startswith("assert") or node.func.attr == "fail"))


def _asserting_with(node):
    """`with self.assertRaises(...)` counts as an assertion."""
    return isinstance(node, ast.With) and any(
        isinstance(i.context_expr, ast.Call)
        and isinstance(i.context_expr.func, ast.Attribute)
        and "assert" in i.context_expr.func.attr
        for i in node.items)


def _literal_nonempty(node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts) > 0
    if isinstance(node, ast.Dict):
        return len(node.keys) > 0
    return False


def _constants(tree):
    """name -> is it bound to a non-empty literal, at module or class level.

    Attribute access is resolved by the bare name, so `self.PLACES` and
    `hardware.KNOWN_GPUS` both find a `PLACES`/`KNOWN_GPUS` assignment. That
    is deliberately loose: this analysis decides whether to STAY SILENT, so
    being generous here costs a missed warning, never a false alarm.
    """
    out = {}

    def scan(body):
        for node in body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = _literal_nonempty(node.value)
            elif isinstance(node, ast.ClassDef):
                scan(node.body)

    scan(tree.body)
    return out


def _cannot_be_empty(node, constants):
    if _literal_nonempty(node):
        return True
    if isinstance(node, ast.Name):
        return constants.get(node.id, False)
    if isinstance(node, ast.Attribute):
        return constants.get(node.attr, False)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("items", "values", "keys"):
            return _cannot_be_empty(func.value, constants)
        if isinstance(func, ast.Name) and func.id == "range":
            first = node.args[0] if node.args else None
            return (isinstance(first, ast.Constant)
                    and isinstance(first.value, int) and first.value > 0)
    return False


def sweep():
    """{"file.py::Class.test_name": "the iterable, as source"} for every test
    whose assertions all sit inside a loop that might not run."""
    found = {}
    for path in sorted(glob.glob(os.path.join(TESTS, "test_*.py"))):
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), path)
        constants = _constants(tree)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]:
                total, risky = 0, []

                def walk(node, loops):
                    nonlocal total
                    for child in ast.iter_child_nodes(node):
                        inner = (loops + [child]
                                 if isinstance(child, (ast.For, ast.AsyncFor)) else loops)
                        if _is_assert(child) or _asserting_with(child):
                            total += 1
                            if inner and not any(_cannot_be_empty(l.iter, constants)
                                                 for l in inner):
                                risky.append(inner[0])
                        walk(child, inner)

                walk(fn, [])
                if total and len(risky) == total:
                    key = "%s::%s.%s" % (os.path.basename(path), cls.name, fn.name)
                    try:
                        found[key] = ast.unparse(risky[0].iter)
                    except Exception:                       # pragma: no cover
                        found[key] = "<unparseable>"
    return found


class TestNoTestQuietlyChecksNothing(unittest.TestCase):

    def test_the_sweep_finds_something_to_look_at(self):
        """The analysis has to be reading the suite. If it returns nothing at
        all the two tests below are vacuous in exactly the way they exist to
        forbid — which would be a fitting way to get this wrong."""
        self.assertTrue(sweep(),
                        "the sweep found no loop-only tests anywhere, which "
                        "means it is not parsing tests/ rather than that the "
                        "suite is clean")

    def test_no_new_test_asserts_only_inside_a_loop_that_may_not_run(self):
        found = sweep()
        new = sorted(set(found) - KNOWN)
        self.assertFalse(
            new,
            "these tests assert ONLY inside a loop whose iterable may be "
            "empty, so they pass when nothing was checked:\n%s\n\n"
            "Add a positive control — assert the collection is non-empty, or "
            "assert something outside the loop that proves the reader found "
            "its subject. Adding the name to KNOWN is accepting the risk."
            % "\n".join("    %s\n        loops over: %s" % (n, found[n])
                        for n in new))

    def test_a_name_that_no_longer_applies_comes_off_the_list(self):
        """The other half, and the reason the list is safe to have. Without
        this it turns into folklore about tests that were fixed years ago, and
        a real one could hide behind a stale name."""
        stale = sorted(KNOWN - set(sweep()))
        self.assertFalse(
            stale,
            "these no longer assert only inside a possibly-empty loop — take "
            "them out of KNOWN:\n%s" % "\n".join("    " + s for s in stale))
