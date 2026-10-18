#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for prewarm_files.py on small temporary ordinary files (no model, no GPU, no root/nice capability).
Run: python3 -m unittest test_prewarm_files"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prewarm_files as pw  # noqa: E402

BIG_MTIME = 1_789_000_000_123_456_789  # 19 digits: not representable exactly as a float


class Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.plan = os.path.join(self.dir, "plan.json")
        self.out = os.path.join(self.dir, "report.json")

    def make(self, name, data, mtime_ns=BIG_MTIME):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(data)
        os.utime(p, ns=(mtime_ns, mtime_ns))
        return p

    def entry(self, role, path, **over):
        st = os.stat(path)
        e = {"role": role, "path": path, "size_bytes": st.st_size, "inode": st.st_ino, "mtime_ns": str(st.st_mtime_ns)}
        e.update(over)
        return e

    def write_plan(self, files, **extra):
        with open(self.plan, "w") as f:
            json.dump({"schema": pw.SCHEMA_IN, "files": files, **extra}, f)

    def go(self):
        code = pw.run(self.plan, self.out)
        with open(self.out) as f:
            return code, json.load(f)


class Success(Tmp):
    def test_all_required_files_read_in_full_with_stat_and_plan_metadata(self):
        small = self.make("a.bin", b"x" * 1000)
        multi = self.make("b.bin", os.urandom(pw.BUF_BYTES * 2 + 17), mtime_ns=BIG_MTIME + 1)  # several buffer refills + a ragged tail
        empty = self.make("c.bin", b"", mtime_ns=0)
        self.write_plan([self.entry("model", multi), self.entry("ple", small), self.entry("opt", empty)],
                        runtime={"tip": "abc"}, source={"n": 3})
        code, rep = self.go()
        self.assertEqual(code, 0)
        self.assertEqual(rep["schema"], pw.SCHEMA_OUT)
        self.assertEqual(rep["status"], "complete")
        self.assertIsNone(rep["failure"])
        self.assertEqual([f["role"] for f in rep["files"]], ["model", "ple", "opt"])
        self.assertTrue(all(f["status"] == "complete" for f in rep["files"]))
        self.assertEqual([f["bytes_read"] for f in rep["files"]], [pw.BUF_BYTES * 2 + 17, 1000, 0])
        self.assertEqual(rep["total_bytes_read"], pw.BUF_BYTES * 2 + 1017)
        for f in rep["files"]:
            self.assertEqual(f["canonical_path"], os.path.realpath(f["path"]))
            self.assertEqual(f["actual_before"], f["actual_after"])
            self.assertGreaterEqual(f["elapsed_s"], 0)
        self.assertEqual(rep["files"][0]["actual_before"]["mtime_ns"], str(BIG_MTIME + 1))
        self.assertEqual(rep["files"][2]["actual_before"]["mtime_ns"], "0")
        self.assertEqual(rep["input_plan"]["runtime"], {"tip": "abc"})
        self.assertEqual(rep["input_plan"]["source"], {"n": 3})
        self.assertEqual(len(rep["input_plan"]["inputs_sha256"]), 64)
        self.assertGreaterEqual(rep["wall_s"], 0)
        self.assertIn("nice", rep)
        self.assertIn("mem_before", rep)
        self.assertIn("mem_after", rep)
        self.assertFalse([n for n in os.listdir(self.dir) if ".tmp-" in n])  # atomic write left nothing behind

    def test_cached_sha256_is_informational_and_never_checked_against_payload(self):
        p = self.make("a.bin", b"payload")
        self.write_plan([self.entry("model", p, sha256="0" * 64)])
        code, rep = self.go()
        self.assertEqual(code, 0)
        self.assertEqual(rep["files"][0]["declared"]["sha256"], "0" * 64)

    def test_symlink_to_regular_file_is_followed_with_target_identity(self):
        target = self.make("real.bin", b"y" * 64)
        link = os.path.join(self.dir, "link.bin")
        os.symlink(target, link)
        self.write_plan([self.entry("model", link)])  # os.stat follows the link
        code, rep = self.go()
        self.assertEqual(code, 0)
        self.assertEqual(rep["files"][0]["path"], link)
        self.assertEqual(rep["files"][0]["canonical_path"], os.path.realpath(target))

    def test_mtime_ns_is_kept_as_exact_19_digit_string(self):
        p = self.make("a.bin", b"1")
        self.assertNotEqual(int(float(BIG_MTIME)), BIG_MTIME)
        self.write_plan([self.entry("model", p)])
        with open(self.plan) as f:
            self.assertIn(f'"mtime_ns": "{BIG_MTIME}"', f.read())
        _, entries = pw.load_plan(self.plan)
        self.assertEqual(entries[0]["mtime_ns"], str(BIG_MTIME))
        self.assertIsInstance(entries[0]["mtime_ns"], str)
        code, rep = self.go()
        self.assertEqual(rep["files"][0]["declared"]["mtime_ns"], str(BIG_MTIME))
        self.assertEqual(rep["files"][0]["actual_before"]["mtime_ns"], str(BIG_MTIME))

    def test_main_returns_exit_code_and_report(self):
        p = self.make("a.bin", b"abc")
        self.write_plan([self.entry("model", p)])
        self.assertEqual(pw.main(["prewarm_files.py", "--inputs", self.plan, "--out", self.out]), 0)
        with open(self.out) as f:
            self.assertEqual(json.load(f)["status"], "complete")


class Failures(Tmp):
    def assert_failed(self, code, rep, expect_code, kind):
        self.assertEqual(code, expect_code)
        self.assertEqual(rep["status"], "failed")
        self.assertEqual(rep["failure"]["kind"], kind)
        return rep

    def test_missing_file_fails_and_later_files_are_reported_not_attempted(self):
        first = self.make("a.bin", b"a" * 10)
        third = self.make("c.bin", b"c" * 10)
        gone = os.path.join(self.dir, "gone.bin")
        self.write_plan([self.entry("first", first),
                         {"role": "second", "path": gone, "size_bytes": 5, "inode": 1, "mtime_ns": "1"},
                         self.entry("third", third)])
        code, rep = self.go()
        self.assert_failed(code, rep, 1, "missing_or_unopenable")
        self.assertEqual([f["status"] for f in rep["files"]], ["complete", "failed", "not_attempted"])
        self.assertEqual(rep["files"][0]["bytes_read"], 10)
        self.assertEqual(rep["files"][2]["bytes_read"], 0)
        self.assertEqual(rep["failure"]["role"], "second")
        self.assertEqual(rep["total_bytes_read"], 10)

    def test_each_identity_field_mismatch_fails_before_reading(self):
        p = self.make("a.bin", b"z" * 100)
        good = self.entry("model", p)
        for key, bad in (("size_bytes", 101), ("inode", good["inode"] + 1), ("mtime_ns", str(BIG_MTIME + 1))):
            with self.subTest(key=key):
                self.write_plan([{**good, key: bad}])
                code, rep = self.go()
                self.assert_failed(code, rep, 1, "identity_mismatch")
                self.assertEqual(rep["files"][0]["bytes_read"], 0)

    def test_non_regular_file_is_refused(self):
        self.write_plan([{"role": "dir", "path": self.dir, "size_bytes": 0, "inode": os.stat(self.dir).st_ino,
                          "mtime_ns": str(os.stat(self.dir).st_mtime_ns)}])
        code, rep = self.go()
        self.assert_failed(code, rep, 1, "not_regular_file")

    def test_short_read_is_detected(self):
        p = self.make("a.bin", b"s" * 100)
        self.write_plan([self.entry("model", p)])
        real = os.readv
        calls = []

        def short(fd, bufs):
            calls.append(1)
            if len(calls) == 1:
                return real(fd, [memoryview(bufs[0])[:40]])  # only 40 of 100 bytes, then EOF
            return 0

        with mock.patch.object(os, "readv", short):
            code, rep = self.go()
        self.assert_failed(code, rep, 1, "short_read")
        self.assertEqual(rep["files"][0]["bytes_read"], 40)

    def test_file_changed_while_being_read_is_detected(self):
        p = self.make("a.bin", b"c" * 100)
        self.write_plan([self.entry("model", p)])
        real = os.readv
        first = []

        def read_then_touch(fd, bufs):
            n = real(fd, bufs)
            if not first:
                first.append(1)
                os.utime(p, ns=(BIG_MTIME + 5, BIG_MTIME + 5))  # same size and inode, new mtime
            return n

        with mock.patch.object(os, "readv", read_then_touch):
            code, rep = self.go()
        self.assert_failed(code, rep, 1, "changed_during_read")
        self.assertEqual(rep["files"][0]["bytes_read"], 100)
        self.assertNotEqual(rep["files"][0]["actual_before"], rep["files"][0]["actual_after"])

    def test_file_that_grows_while_being_read_is_detected(self):
        p = self.make("a.bin", b"g" * 100)
        self.write_plan([self.entry("model", p)])
        real = os.readv
        first = []

        def read_then_append(fd, bufs):
            n = real(fd, bufs)
            if not first:
                first.append(1)
                with open(p, "ab") as f:
                    f.write(b"more")
            return n

        with mock.patch.object(os, "readv", read_then_append):
            code, rep = self.go()
        self.assert_failed(code, rep, 1, "grew_during_read")

    def test_path_replaced_during_read_is_detected(self):
        p = self.make("a.bin", b"r" * 100)
        self.write_plan([self.entry("model", p)])
        real = os.readv
        first = []

        def read_then_replace(fd, bufs):
            n = real(fd, bufs)
            if not first:
                first.append(1)
                other = self.make("other.bin", b"o" * 100)
                os.replace(other, p)  # the opened descriptor still names the old inode
            return n

        with mock.patch.object(os, "readv", read_then_replace):
            code, rep = self.go()
        self.assert_failed(code, rep, 1, "path_replaced_during_read")

    def test_partial_failure_never_reports_success_for_skipped_files(self):
        a = self.make("a.bin", b"a" * 8)
        b = self.make("b.bin", b"b" * 8)
        self.write_plan([self.entry("a", a), self.entry("b", b, size_bytes=9)])
        code, rep = self.go()
        self.assertEqual(code, 1)
        self.assertNotEqual(rep["status"], "complete")
        self.assertEqual({f["role"]: f["status"] for f in rep["files"]}, {"a": "complete", "b": "failed"})

    def test_incomplete_receipt_is_written_before_work_starts(self):
        p = self.make("a.bin", b"a")
        self.write_plan([self.entry("model", p)])
        seen = []
        real = pw.warm_file

        def spy(entry, buf):
            with open(self.out) as f:
                seen.append(json.load(f)["status"])
            return real(entry, buf)

        with mock.patch.object(pw, "warm_file", spy):
            self.assertEqual(pw.run(self.plan, self.out), 0)
        self.assertEqual(seen, ["incomplete"])


class BadPlan(Tmp):
    def bad(self, plan_obj):
        with open(self.plan, "w") as f:
            if isinstance(plan_obj, str):
                f.write(plan_obj)
            else:
                json.dump(plan_obj, f)
        code, rep = self.go()
        self.assertEqual(code, 2)
        self.assertEqual(rep["status"], "failed")
        self.assertEqual(rep["failure"]["kind"], "bad_plan")
        self.assertEqual(rep["files"], [])
        return rep

    def good(self):
        p = self.make("a.bin", b"a")
        return self.entry("model", p)

    def test_plan_shape_errors_fail_with_receipt_and_no_reads(self):
        e = self.good()
        self.bad("not json")
        self.bad({"schema": "other", "files": [e]})
        self.bad({"schema": pw.SCHEMA_IN})
        self.bad({"schema": pw.SCHEMA_IN, "files": []})
        self.bad({"schema": pw.SCHEMA_IN, "files": [e, dict(e)]})  # duplicate role
        self.bad({"schema": pw.SCHEMA_IN, "files": ["nope"]})

    def test_mtime_ns_must_be_exact_decimal_string(self):
        e = self.good()
        for bad in (BIG_MTIME, float(BIG_MTIME), "1.5e18", "-1", " 12", "012", "", "1_000", "١٢٣", None):
            with self.subTest(mtime=bad):
                rep = self.bad({"schema": pw.SCHEMA_IN, "files": [{**e, "mtime_ns": bad}]})
                self.assertIn("mtime_ns", rep["failure"]["message"])
        del e["mtime_ns"]
        self.bad({"schema": pw.SCHEMA_IN, "files": [e]})

    def test_other_field_types_are_strict(self):
        e = self.good()
        for key, bad in (("size_bytes", "1"), ("size_bytes", 1.0), ("size_bytes", True), ("inode", -1), ("inode", None),
                         ("role", ""), ("path", 5), ("sha256", "abc")):
            with self.subTest(key=key, bad=bad):
                self.bad({"schema": pw.SCHEMA_IN, "files": [{**e, key: bad}]})

    def test_missing_plan_file_is_a_bad_plan(self):
        os.path.exists(self.plan) and os.remove(self.plan)
        code, rep = self.go()
        self.assertEqual(code, 2)
        self.assertEqual(rep["failure"]["kind"], "bad_plan")


if __name__ == "__main__":
    unittest.main()
