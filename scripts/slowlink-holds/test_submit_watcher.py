#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of run-with-watcher.sh using a fake journalctl, fake pgrep and fake fabric command in a temp dir.
No live journalctl, GPU, model or fabric is ever run. Tests signal only processes THEY started, by recorded PID/PGID.

    python3 scripts/slowlink-holds/test_submit_watcher.py -v
"""
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "run-with-watcher.sh")

FAKE_JOURNALCTL = """#!/usr/bin/env bash
printf '%s\\n' "$@" >"$FAKE_DIR/journalctl.args"
ls /proc/$$/fd >"$FAKE_DIR/journalctl.fds"
echo "$$" >"$FAKE_DIR/journalctl.pid"
echo started >"$FAKE_DIR/journalctl.started"
case "${FAKE_JOURNAL_MODE:-ok}" in
  dead) exit 1 ;;
  ignoreterm) trap '' TERM ;;
esac
exec sleep 300
"""

FAKE_PGREP = """#!/usr/bin/env bash
printf '%s\\n' "$@" >"$FAKE_DIR/pgrep.args"
echo "4242 journalctl -k -f -n 0 (fixture)"
"""

FAKE_FABRIC = """#!/usr/bin/env bash
D=$FAKE_DIR
printf '%s\\n' "$@" >"$D/fabric.argv"
ls /proc/$$/fd >"$D/fabric.fds"
read -r wpid <"$OUT_PREFIX.watcher.pgid"
{
  echo "$wpid"
  kill -0 -- "-$wpid" && echo live
  ps -o pgid= -o stat= -p "$wpid"
  [ -e "$D/journalctl.started" ] && echo started
} >"$D/fabric.watcher" 2>&1
echo out-line
echo err-line >&2
if [ -n "${FAKE_FABRIC_SLEEP:-}" ]; then
  touch "$D/fabric.ready"
  sleep "$FAKE_FABRIC_SLEEP"
fi
exit "${FAKE_FABRIC_RC:-0}"
"""


def fd_set(path):
    with open(path) as f:
        return set(f.read().split()) - {"255"}  # 255 = bash's own script fd


def pgid_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def read(path):
    with open(path) as f:
        return f.read()


class WatcherWrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.fake = os.path.join(root, "fake")
        self.bindir = os.path.join(root, "bin")
        self.out = os.path.join(root, "out")
        for d in (self.fake, self.bindir, self.out):
            os.mkdir(d)
        self.fabric = os.path.join(root, "fake-fabric.sh")
        for path, body in ((os.path.join(self.bindir, "journalctl"), FAKE_JOURNALCTL),
                           (os.path.join(self.bindir, "pgrep"), FAKE_PGREP),
                           (self.fabric, FAKE_FABRIC)):
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, 0o755)
        self.procs = []
        self.addCleanup(self.reap_own)

    def reap_own(self):
        # Only processes this test started: wrapper PGID (== its pid, own session) and the watcher PGID it recorded.
        for proc, prefix in self.procs:
            for pgid in (proc.pid, self.recorded_pgid(prefix)):
                if pgid:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()

    @staticmethod
    def recorded_pgid(prefix):
        if prefix is None:
            return None
        try:
            return int(read(prefix + ".watcher.pgid"))
        except (OSError, ValueError):
            return None

    def start(self, prefix, argv, env=None, pass_fds=()):
        e = dict(os.environ)
        e.update(PATH=self.bindir + os.pathsep + e["PATH"], FAKE_DIR=self.fake)
        e.pop("OUT_PREFIX", None)
        if prefix is not None:
            e["OUT_PREFIX"] = prefix
        e.update(env or {})
        proc = subprocess.Popen(
            ["bash", SCRIPT, *argv], env=e, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, pass_fds=pass_fds,
            preexec_fn=lambda: [signal.signal(s, signal.SIG_DFL) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)])
        self.procs.append((proc, prefix))
        return proc

    def run_wrapper(self, prefix, argv, **kw):
        proc = self.start(prefix, argv, **kw)
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            self.fail("wrapper did not finish")
        return proc.returncode, out, err

    def prefix(self, name="r7"):
        return os.path.join(self.out, name)

    def fake_file(self, name):
        return os.path.join(self.fake, name)

    def assert_watcher_gone(self, prefix):
        pgid = self.recorded_pgid(prefix)
        self.assertIsNotNone(pgid)
        self.assertFalse(pgid_alive(pgid), "watcher group still alive")

    def test_success_status_ownership_and_fds(self):
        prefix = os.path.join(self.out, "new", "sub", "r7")  # parent is created after validation
        argv = [self.fabric, "--flag", "a b", "c'd"]
        extra = os.open(os.path.join(self.tmp.name, "extra"), os.O_WRONLY | os.O_CREAT)
        self.addCleanup(os.close, extra)
        rc, _, _ = self.run_wrapper(prefix, argv, env={"FAKE_FABRIC_RC": "7"}, pass_fds=(extra,))
        self.assertEqual(rc, 7)
        self.assertEqual(read(prefix + ".exit"), "7\n")
        log = read(prefix + ".fabric.log")
        self.assertIn("out-line", log)
        self.assertIn("err-line", log)
        self.assertEqual(shlex.split(read(prefix + ".command.txt")), argv)
        self.assertEqual(read(self.fake_file("fabric.argv")).split("\n")[:-1], argv[1:])
        # watcher: same flags as the approved pattern, no fd > 2 (control: the fabric command DOES see the extra fd)
        self.assertEqual(read(self.fake_file("journalctl.args")).split(), ["-k", "-f", "-n", "0", "-o", "short-iso", "--no-pager"])
        self.assertEqual(fd_set(self.fake_file("journalctl.fds")), {"0", "1", "2"})
        self.assertIn(str(extra), fd_set(self.fake_file("fabric.fds")))
        # own PGID live (pgid == pid, not zombie) before the fabric command started, distinct from the wrapper group
        wpid, live, state, started = read(self.fake_file("fabric.watcher")).split("\n")[:4]
        self.assertEqual(wpid, read(prefix + ".watcher.pgid").strip())
        self.assertEqual(live, "live")
        self.assertEqual(state.split()[0], wpid)
        self.assertFalse(state.split()[1].startswith("Z"))
        self.assertEqual(started, "started")
        self.assertEqual(wpid, read(self.fake_file("journalctl.pid")).strip())
        self.assertNotEqual(int(wpid), self.procs[-1][0].pid)
        # cleanup: own group gone, diagnostics recorded only
        self.assert_watcher_gone(prefix)
        self.assertEqual(read(self.fake_file("pgrep.args")).split("\n")[:-1], ["-af", "[j]ournalctl -k -f"])
        self.assertIn("4242", read(prefix + ".watchers-after.txt"))
        self.assertTrue(os.path.exists(prefix + ".dmesg.log"))

    def test_dead_watcher_prevents_command(self):
        prefix = self.prefix()
        rc, _, err = self.run_wrapper(prefix, [self.fabric], env={"FAKE_JOURNAL_MODE": "dead"})
        self.assertEqual(rc, 2)
        self.assertIn("watcher", err)
        self.assertFalse(os.path.exists(self.fake_file("fabric.argv")))
        self.assertFalse(os.path.exists(prefix + ".command.txt"))
        self.assertFalse(os.path.exists(prefix + ".fabric.log"))
        self.assertEqual(read(prefix + ".exit"), "2\n")

    def test_refuses_reuse_without_writing(self):
        for suffix in ("", ".exit", ".fabric.log", ".dmesg.log", ".watcher.pgid", ".watchers-after.txt", ".command.txt"):
            with self.subTest(suffix=suffix):
                prefix = self.prefix("reuse" + (suffix.replace(".", "_") or "_prefix"))
                with open(prefix + suffix, "w") as f:
                    f.write("keep")
                before = sorted(os.listdir(self.out))
                rc, _, err = self.run_wrapper(prefix, [self.fabric])
                self.assertEqual(rc, 2)
                self.assertIn("Refusing", err)
                self.assertEqual(read(prefix + suffix), "keep")
                self.assertEqual(sorted(os.listdir(self.out)), before)
                self.assertFalse(os.path.exists(self.fake_file("journalctl.started")))
                self.assertFalse(os.path.exists(self.fake_file("fabric.argv")))

    def test_input_validation_writes_nothing(self):
        newparent = os.path.join(self.out, "never")
        cases = [
            (None, [self.fabric]),
            ("relative/prefix", [self.fabric]),
            (os.path.join(newparent, "r7"), []),
            (os.path.join(newparent, "r7") + "/", [self.fabric]),
        ]
        for prefix, argv in cases:
            with self.subTest(prefix=prefix, argv=argv):
                rc, _, _ = self.run_wrapper(prefix, argv)
                self.assertEqual(rc, 2)
                self.assertFalse(os.path.exists(newparent))
                self.assertEqual(os.listdir(self.out), [])
                self.assertFalse(os.path.exists(self.fake_file("journalctl.started")))

    def test_signals_record_status_and_clean_up_own_group(self):
        for name, sig, expect in (("TERM", signal.SIGTERM, 143), ("INT", signal.SIGINT, 130), ("HUP", signal.SIGHUP, 129)):
            with self.subTest(sig=name):
                prefix = self.prefix("sig" + name)
                ready = self.fake_file("fabric.ready")
                if os.path.exists(ready):
                    os.unlink(ready)
                proc = self.start(prefix, [self.fabric], env={"FAKE_FABRIC_SLEEP": "2"})
                deadline = time.time() + 30
                while not os.path.exists(ready):
                    self.assertLess(time.time(), deadline, "fake fabric never started")
                    self.assertIsNone(proc.poll())
                    time.sleep(0.05)
                os.kill(proc.pid, sig)  # wrapper only; trap runs once the foreground command returns
                proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, expect)
                self.assertEqual(read(prefix + ".exit"), "%d\n" % expect)
                self.assertIn("out-line", read(prefix + ".fabric.log"))
                self.assert_watcher_gone(prefix)
                self.assertTrue(os.path.exists(prefix + ".watchers-after.txt"))

    def test_term_ignoring_watcher_is_killed_after_bounded_wait(self):
        prefix = self.prefix()
        t0 = time.time()
        rc, _, _ = self.run_wrapper(prefix, [self.fabric], env={"FAKE_JOURNAL_MODE": "ignoreterm"})
        self.assertEqual(rc, 0)
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(read(prefix + ".exit"), "0\n")
        self.assert_watcher_gone(prefix)


class WatcherWrapperSourceTest(unittest.TestCase):
    def setUp(self):
        self.src = read(SCRIPT)

    def test_syntax(self):
        subprocess.run(["bash", "-n", SCRIPT], check=True)

    def test_no_broad_kill_and_no_hardcoded_target(self):
        self.assertIsNone(re.search(r"pkill|killall|xargs", self.src))
        for line in self.src.splitlines():
            if re.search(r"\bkill\b", line) and not line.lstrip().startswith("#"):
                self.assertIn('"-$WPID"', line)
        for word in ("fabric-run", "xtx", "ROCR", "HIP_VISIBLE", "/home/kaden", "--mem-gb", "retry"):
            self.assertNotIn(word, self.src)

    def test_watcher_starts_before_command_in_own_session(self):
        self.assertIn("setsid bash -c", self.src)
        self.assertIn("journalctl -k -f -n 0 -o short-iso --no-pager", self.src)
        self.assertIn("set +m", self.src)
        self.assertLess(self.src.index("setsid bash -c"), self.src.index('"${CMD[@]}" >'))
        self.assertLess(self.src.index("trap finish EXIT"), self.src.index("setsid bash -c"))
        self.assertLess(self.src.index("Refusing"), self.src.index("mkdir -p"))


if __name__ == "__main__":
    unittest.main()
