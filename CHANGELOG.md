## [0.3.x] - 2026-07-22
These changes address compatibility issue introduced by `vLLM`, as well as formatting and squashing of annoying warnings. The requirements and package dependencies have been updated accordingly.
The functionality remains the same as before, but with real speedups due to batch processing with `vLLM`!

### Added
 - To maximize efficiency, `privatize` now also accepts a list of `epsilon`, which will then returning a dictionary of privatized texts for each privacy budget. This runs in nearly the same amount of time as single mode!
 - Cleanup function for vLLM, to allow for repeat runs on different parameter setups.

## [0.2.0] - 2026-07-14
### Added
 - DP-ST now runs quicker! We have shifted from `transformers` to `vLLM`, which runs the reconstruction stage in batches. Note that the dependencies for this package have now changed.

### Changed
 - By default, we now run reconstruction with a temperature value of `0.1`. This differs from the `0` value of the original work.

---

## [0.1.0]
Version history starts here!
