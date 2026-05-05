# PH6 Dual-Speed SoSo/TOK Stress Proof

Status: PASS

## Test

Command:

python3 ph6_full_stack_coherence.py \
  --source oracle \
  --frames 300 \
  --dual-speed-soso \
  --soso-fast \
  --tok-fast \
  --soso-slow-delay-ms 500 \
  --tok-slow-delay-ms 500 \
  --allow-lane2-backlog \
  --run-replay

## Result

- 500ms per frame slow-path delay.
- CRAM/PSEUDO completed all 300 frames.
- blocked_by_lane2: false.
- Replay: PASS.
- Hash chain: valid.
- final_verdict: PASS.

## Conclusion

PH6 preserves truth even when cognition falls behind.

CRAM + PSEUDO = hard real-time authority path.
SoSo-FAST + TOK-FAST = immediate advisory shadow path.
SoSo-SLOW + TOK-SLOW = delayed cognition path.

FAST may observe now.
SLOW may understand later.
Neither may decide truth.
