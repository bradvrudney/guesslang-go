package guesslang_test

import (
	"fmt"

	guesslang "github.com/bradvrudney/guesslang-go"
)

func ExampleGuess() {
	code := "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n"
	lang, ok := guesslang.Guess(code)
	fmt.Println(lang, ok)
	// Output: Go true
}

func ExampleProbabilities() {
	scores, _ := guesslang.Probabilities("SELECT id FROM users\nWHERE id = 1;\n")
	fmt.Println(len(scores), scores[0].Language)
	// Output: 54 SQL
}
