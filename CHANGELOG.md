## [0.2.0] - 2026-07-014
### Added
DP-ST now runs quicker! We have shifted from `transformers` to `vLLM`, which runs the reconstruction stage in batches. Note that the dependencies for this package have now changed.

### Changed
By default, we now run reconstruction with a temperature value of `0.1`. This differs from the `0` value of the original work.

---

## [0.1.0]
Version history starts here!
