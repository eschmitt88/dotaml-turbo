# RETRACTED — root cause was NOT PyTorch

The upstream report at `ISSUE_BODY.md` / `torch-dataloader-gc-segfault-report.md`
and the synthetic reproducer at `torch-dataloader-gc-segfault-repro.py` were
filed as pytorch/pytorch#184062 on 2026-05-17. The report attributed
DataLoader-fetch-during-GC segfaults on an RTX 5080 to a torch tensor-GC bug.

**That diagnosis was wrong.** A subsequent hardware investigation
(2026-05-21) identified the true root cause: silent RAM bit-flips from
unstable DDR5 EXPO 6000 MT/s on non-ECC memory. With EXPO disabled
(JEDEC 4800 MT/s) the same workload runs clean.

Retraction posted to the upstream issue:
https://github.com/pytorch/pytorch/issues/184062#issuecomment-4508610051

Full investigation log at
`_meta/hardware-investigation-2026-05-21/investigation.log`.

The original report files in this directory are kept for posterity;
do NOT use them as guidance for future torch / Blackwell debugging.
