# guesslang-go

[![CI](https://github.com/bradvrudney/guesslang-go/actions/workflows/ci.yml/badge.svg)](https://github.com/bradvrudney/guesslang-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/bradvrudney/guesslang-go.svg)](https://pkg.go.dev/github.com/bradvrudney/guesslang-go)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Guess the programming language of a piece of source code, in pure Go.

A port of the inference half of [guesslang](https://github.com/yoeo/guesslang)
(MIT, by Y. Somda): the same trained model, run without Python or TensorFlow.
The 2.7 MB model is embedded in the package, so there is nothing to download
or install at run time.

```go
import "github.com/bradvrudney/guesslang-go"

lang, ok := guesslang.Guess(text)        // "Python", true
scores, _ := guesslang.Probabilities(text) // all 54 languages, most probable first
```

- `Guess` returns the top language and whether guesslang's own rule finds it
  reliable (the top probability is more than two standard deviations above
  the mean of all of them).
- `Probabilities` gives the whole answer, for a caller that wants its own bar.
  The model always names a language, even for ordinary writing, so a program
  that colours text should also ask for a minimum probability and a minimum
  length.
- `Languages` lists the 54 languages: Assembly, Batchfile, C, C#, C++,
  Clojure, CMake, COBOL, CoffeeScript, CSS, CSV, Dart, DM, Dockerfile,
  Elixir, Erlang, Fortran, Go, Groovy, Haskell, HTML, INI, Java, JavaScript,
  JSON, Julia, Kotlin, Lisp, Lua, Makefile, Markdown, Matlab, Objective-C,
  OCaml, Pascal, Perl, PHP, PowerShell, Prolog, Python, R, Ruby, Rust, Scala,
  Shell, SQL, Swift, TeX, TOML, TypeScript, Verilog, Visual Basic, XML, YAML.

## How it works

The model is a TensorFlow `DNNLinearCombinedClassifier`. A text is cut into
every pair of neighbouring bytes (the first 10,000 pairs), each pair is
hashed (FarmHash Fingerprint64) into one of 5,000 buckets, and the buckets
feed two parts that are added and turned into probabilities by a softmax: the
mean of their 70-number embeddings through two ReLU layers (512 and 32 units),
and the sum of a per-bucket weight for each language.

## Checked against the original

`testdata/reference.json` holds the probabilities the original guesslang
(2.2.1 with TensorFlow 2.5.0) gives for the 32 files in `testdata/samples`
(code in many languages, ordinary writing, non-ASCII text, one tiny, one
over 10,000 bytes). The test requires every one of the 54 probabilities of
every sample to agree within 0.0001; the largest difference is about 2e-7.

`tools/regenerate.sh` rebuilds the embedded model and the reference answers
from the original, in Docker.

## Contributing

Bug reports, tests and documentation are welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md). Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md). Security problems go to
[SECURITY.md](SECURITY.md).

## Licence

MIT. The model and the language list are guesslang's, copyright Y. Somda.
