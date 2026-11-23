package guesslang

import farm "github.com/dgryski/go-farm"

const (
	// maxTokens is how many two-byte tokens of a text the model reads.
	maxTokens = 10000
	// buckets is how many ids the tokens are hashed into.
	buckets = 5000
)

// bucketsOf turns a text into the ids the model reads: every pair of
// neighbouring bytes, written as the two bytes with a space between (what
// TensorFlow's bytes_split and ngrams make of it), is hashed with
// FarmHash's Fingerprint64 and taken modulo the number of buckets. Only
// the first maxTokens pairs count, and a text of fewer than two bytes has
// none.
func bucketsOf(text string) []int {
	n := len(text) - 1
	if n <= 0 {
		return nil
	}
	if n > maxTokens {
		n = maxTokens
	}
	ids := make([]int, n)
	var tok [3]byte
	tok[1] = ' '
	for i := 0; i < n; i++ {
		tok[0], tok[2] = text[i], text[i+1]
		ids[i] = int(farm.Fingerprint64(tok[:]) % buckets)
	}
	return ids
}
