package guesslang

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type refEntry struct {
	Probabilities [][2]any `json:"probabilities"`
	LanguageName  *string  `json:"language_name"`
}

func loadReference(t *testing.T) map[string]refEntry {
	t.Helper()
	b, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref map[string]refEntry
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	return ref
}

// TestMatchesTheOriginal compares every probability with what the original
// guesslang (TensorFlow 2.5.0, version 2.2.1) gave for the same sample.
func TestMatchesTheOriginal(t *testing.T) {
	ref := loadReference(t)
	if len(ref) < 30 {
		t.Fatalf("only %d reference samples", len(ref))
	}
	worst := 0.0
	for name, want := range ref {
		text, err := os.ReadFile(filepath.Join("testdata", "samples", name))
		if err != nil {
			t.Fatal(err)
		}
		got, err := Probabilities(string(text))
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want.Probabilities) {
			t.Fatalf("%s: %d scores, want %d", name, len(got), len(want.Probabilities))
		}
		wantByLang := map[string]float64{}
		for _, p := range want.Probabilities {
			wantByLang[p[0].(string)] = p[1].(float64)
		}
		for _, s := range got {
			d := math.Abs(s.Probability - wantByLang[s.Language])
			worst = math.Max(worst, d)
			if d > 1e-4 {
				t.Errorf("%s: %s is %.6f, the original says %.6f", name, s.Language, s.Probability, wantByLang[s.Language])
			}
		}
		if got[0].Language != want.Probabilities[0][0].(string) {
			t.Errorf("%s: most probable is %s, the original says %s", name, got[0].Language, want.Probabilities[0][0])
		}
	}
	t.Logf("largest difference from the original: %g", worst)
}

func TestGuessFollowsTheOriginalsRule(t *testing.T) {
	for name, want := range loadReference(t) {
		text, _ := os.ReadFile(filepath.Join("testdata", "samples", name))
		got, ok := Guess(string(text))
		switch {
		case want.LanguageName == nil && ok:
			t.Errorf("%s: %q, the original found nothing reliable", name, got)
		case want.LanguageName != nil && (!ok || got != *want.LanguageName):
			t.Errorf("%s: %q (%v), the original says %q", name, got, ok, *want.LanguageName)
		}
	}
}

func TestEmptyAndTinyTexts(t *testing.T) {
	for _, s := range []string{"", "   \n\t", "a"} {
		if got, ok := Guess(s); ok && strings.TrimSpace(s) == "" {
			t.Errorf("Guess(%q) = %q", s, got)
		}
		if _, err := Probabilities(s); err != nil {
			t.Errorf("Probabilities(%q): %v", s, err)
		}
	}
}

func TestProbabilitiesSumToOneAndAreSorted(t *testing.T) {
	scores, err := Probabilities("package main\n\nfunc main() {}\n")
	if err != nil || len(scores) != 54 || len(Languages()) != 54 {
		t.Fatalf("%d scores, %d languages, err %v", len(scores), len(Languages()), err)
	}
	var sum float64
	for i, s := range scores {
		sum += s.Probability
		if i > 0 && s.Probability > scores[i-1].Probability {
			t.Fatal("not sorted")
		}
	}
	if math.Abs(sum-1) > 1e-6 || scores[0].Language != "Go" {
		t.Fatalf("sum %v, top %s", sum, scores[0].Language)
	}
}

func TestSafeForConcurrentUse(t *testing.T) {
	done := make(chan bool)
	for i := 0; i < 8; i++ {
		go func() {
			Guess("def f(x):\n    return x + 1\n")
			done <- true
		}()
	}
	for i := 0; i < 8; i++ {
		<-done
	}
}

func BenchmarkGuessSmallSnippet(b *testing.B) {
	text := "def f(x):\n    for i in range(x):\n        print(i)\n"
	for i := 0; i < b.N; i++ {
		Guess(text)
	}
}
