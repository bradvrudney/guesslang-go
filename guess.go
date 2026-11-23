// Package guesslang guesses the programming language of source code, in
// pure Go. It is a port of the inference half of guesslang
// (https://github.com/yoeo/guesslang, MIT): the same trained model, embedded
// in the program, run without Python or TensorFlow.
//
// The model knows 54 languages and always names one, even for text that is
// not code. Probabilities exposes the whole answer so that a caller can set
// its own bar; Guess applies guesslang's own rule for when to trust it.
package guesslang

import (
	"math"
	"strings"
)

// Languages lists the languages the model knows.
func Languages() []string {
	nw, err := model()
	if err != nil {
		return nil
	}
	return append([]string(nil), nw.languages...)
}

// Guess returns the most probable language of text, and whether guesslang's
// rule finds that answer reliable: the probability of the top language must
// be more than two standard deviations above the mean of all of them. It
// returns false for empty text.
func Guess(text string) (string, bool) {
	if strings.TrimSpace(text) == "" {
		return "", false
	}
	scores, err := Probabilities(text)
	if err != nil || len(scores) < 2 {
		return "", false
	}
	if !reliable(scores) {
		return "", false
	}
	return scores[0].Language, true
}

// reliable is guesslang's rule, with the sample standard deviation that
// Python's statistics.stdev computes.
func reliable(scores []Score) bool {
	var mean float64
	for _, s := range scores {
		mean += s.Probability
	}
	mean /= float64(len(scores))
	var ss float64
	for _, s := range scores {
		ss += (s.Probability - mean) * (s.Probability - mean)
	}
	sd := math.Sqrt(ss / float64(len(scores)-1))
	return scores[0].Probability > mean+2*sd
}
