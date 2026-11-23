# guesslang-go

A pure-Go port of the inference half of [guesslang](https://github.com/yoeo/guesslang)
(MIT): the same trained model, embedded, run without Python or TensorFlow.
Public repository: only commit things that are fine to publish.

## The one rule

Its answers must match the original guesslang 2.2.1 (TensorFlow 2.5.0).
`TestMatchesTheOriginal` compares all 54 probabilities for every file in
`testdata/samples` with `testdata/reference.json` (tolerance 1e-4; the real
difference is about 2e-7). `TestGuessFollowsTheOriginalsRule` compares `Guess`.
A change that moves those numbers is wrong unless it deliberately changes what
the package promises; say why in the commit.

## Layout

- `tokens.go`: text to bucket ids (every pair of neighbouring bytes, written
  `a b`, FarmHash Fingerprint64 mod 5000; first 10,000 pairs only).
- `model.go`: loads `model/model.bin` (embedded) and runs the network: mean
  embedding, two ReLU layers, plus the linear part, softmax.
- `guess.go`: `Guess` (guesslang's reliability rule: top probability more than
  two sample standard deviations above the mean), `Languages`.
- `model/model.bin`, `model/languages.json`: generated or copied from the
  original; the output order of the model is the key order of `languages.json`.
- `tools/`: `extract.py` and `pack.py` build the model file and reference
  answers; `regenerate.sh` runs them in Docker. `model/weights.json` is an
  intermediate (15 MB, git-ignored).

## Working here

- Go 1.22 is the minimum (`go.mod`); CI runs 1.22 and the latest.
- Before committing: `gofmt -l .` prints nothing, `go vet ./...`,
  `go test -race ./...`.
- Adding a reference sample: put the file in `testdata/samples/` (text you
  wrote or that is free to use, never private code), run `tools/regenerate.sh`
  (needs Docker), commit the sample and the new `reference.json`. `model.bin`
  must come out byte for byte the same.
- Never run the original or fetch anything from the package at run time: the
  package makes no network requests and writes no files.
- Update `CHANGELOG.md` for anything a user of the package would notice.

## Known limits (do not "fix" without evidence)

- The model always names a language, even for ordinary writing, and is
  unreliable under about four lines. Callers set their own bar from
  `Probabilities`; `Guess` is only the original's rule.
- Floating-point sums are in a different order from TensorFlow's, so values
  can differ in the 7th decimal and, on very long texts, up to about 3e-4.
