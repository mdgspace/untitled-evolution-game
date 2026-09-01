# Open-loop controller validation

- Unity: 6000.5.4f1, Windows Editor, SampleScene.
- v9 EditMode job `d1bcea84d5574d3b856886ad7a1e3689`: 14 passed, 0 failed, 0 skipped.
- v9 PlayMode job `fb50d958af844027ae0f1a8694c9eb64`: 2 passed, 0 failed, 0 skipped in 52.27 seconds.
- v9 scaled long-run result: `tick_delta=14197 population=6 replacements=1 m3_updates=4205 action_magnitude_variants=9 bias=0 m2_loss=117.2036 m3_loss=1.37310588 control_failures=0`.
- The v9 cadence/topology invalidates v8 state: it uses schema 9 and the `native-v9-openloop-h5-a4-ff32-m2h6-mlp5-m3h8-t2-mlp5-localpos-m2goal-m3direct10-antizero4-bias075t13333` fingerprint. At startup, exact v6/v7/v8 primary, backup, and temporary checkpoints are removed before the v9 checkpoint path is opened.
- Live Profiler sample: CPU frame 10.9528 ms, main thread 5.6549 ms, render thread 2.6402 ms, GPU 0.27648 ms.
- Live memory sample: 22,805 bytes across 332 managed allocations in the sampled frame; GC used memory 1,119,891,456 bytes; 2D physics memory 1,106,456 bytes.

The Profiler sample includes Unity Editor and installed package overhead. It is evidence of the observed editor run, not a standalone-player performance budget. The per-frame managed allocations remain an optimization target.
