# Contributing

Thank you for helping. This is a small package with one job, so most useful
contributions are bug reports, better tests and documentation.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## What the package is

A port of the inference half of [guesslang](https://github.com/yoeo/guesslang):
the same trained model, run in Go. **Its answers should match the original's.**
That is what `TestMatchesTheOriginal` checks, against probabilities the
original gave for the files in `testdata/samples`. A change that moves those
numbers needs a reason in the pull request.

Retraining the model, or adding languages, belongs upstream in guesslang.

## Setting up

You need Go 1.22 or later. Nothing else: no Python, no TensorFlow.

```sh
git clone https://github.com/bradvrudney/guesslang-go
cd guesslang-go
go test -race ./...
```

Before sending a pull request:

```sh
gofmt -l .        # prints nothing
go vet ./...
go test -race ./...
```

CI runs the same on Go 1.22 and the latest Go.

## Reporting a bug

Use the bug template. The most useful report is the smallest text that gets a
wrong answer, what you expected, and, if you can run it, what the original
guesslang says for the same text. Only share text you are free to publish.

## Sending a change

1. Open an issue first for anything bigger than a small fix, so the work is
   not wasted.
2. Keep one change to a pull request, with a test that fails without it.
3. Write comments and commit messages that say why.

## Adding a sample to the reference set

`testdata/samples/` holds the texts and `testdata/reference.json` the
original's answers for them. To add a sample, put the file in
`testdata/samples/` (text you wrote or that is clearly free to use, never
somebody's private code) and run `tools/regenerate.sh`, which needs Docker and
runs the original guesslang (2.2.1, TensorFlow 2.5.0). Commit the new sample
and the updated `reference.json`. The embedded model (`model/model.bin`)
should come out byte for byte the same; if it does not, say so in the pull
request.

## Licence

By contributing you agree that your work is released under the MIT licence in
[LICENSE](LICENSE).
